"""
analyzer.py -- Sentiment analysis and purpose classification.

Given clean article text, produces:
  * VADER sentiment metrics (positive / neutral / negative / compound)
  * Three calibrated scores in 0-100: informational, harmful, agenda_based
  * A verdict (dominant class) with a confidence margin
  * Evidence: the specific sentences that drove each score
  * Article stats: reading ease, reading time, sentence counts

No paid AI services: everything runs locally with VADER + evidence-weighted
lexicons built for news language specifically.
"""

import math
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_VADER = SentimentIntensityAnalyzer()

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\u201c(])")
PARA_SPLIT = re.compile(r"\n\s*\n")

# ---------------------------------------------------------------- lexicons --

# Terms tied to harm: violence, disaster, distress.
HARM_TERMS = {
    "attack", "attacked", "killed", "killing", "dead", "death", "died", "massacre",
    "bomb", "bombing", "explosion", "blast", "shooting", "gunman", "hostage",
    "war", "assault", "violence", "casualtie", "casualties", "injur", "injured",
    "victim", "victims", "terror", "terrorism", "terrorist", "abuse", "torture",
    "catastrophe", "catastrophic", "disaster", "tragedy", "tragic", "crisis",
    "famine", "epidemic", "outbreak", "pandemic", "collapse", "crashed",
    "drown", "murder", "stabbing", "behead", "missile", "strike", "airstrike",
    "raid", "threat", "menace", "devastat", "devastating", "disturb", "horrific",
    "horror", "panic", "wounded", "blood", "fatal", "slaughter", "genocide",
}

# Terms that push a viewpoint: persuasion, emotive framing, conspiracy framing.
AGENDA_TERMS = {
    "must", "should not be allowed", "clearly", "obviously", "undoubtedly",
    "undeniably", "without a doubt", "the truth is", "everyone knows",
    "wake up", "shocking", "outrageous", "disgrace", "disgusting", "evil",
    "corrupt", "corruption", "puppet", "propaganda", "nationalist", "regime",
    "elites", "elite", "betray", "betrayal", "foot soldier", "traitor",
    "lie", "liar", "lies", "agenda", "hidden", "exposed", "unmask",
    "they don't want", "media won't", "you won't believe", "wake the hate up",
    "radical", "extremist", "fanatic", "anti-national", "steroid",
    "destroy", "crush", "humiliate", "shameful", "infamous", "menacing",
    "conspiracy", "cover-up", "whitewash", "silence", "censor", "censored",
}

# Markers of factual, sourced reporting.
SOURCE_TERMS = {
    "according to", "reported", "said", "stated", "spokesperson", "officials said",
    "announced", "confirmed", "data", "statistics", "study", "survey",
    "reuters", "associated press", "afp", "pti", "ansa", "bloomberg",
    "yesterday", "on monday", "on tuesday", "on wednesday", "on thursday",
    "on friday", "on saturday", "on sunday", "testified", "filed", "documented",
    "press conference", "statement", "warrant", "court", "tribunal", "filing",
}

# Formulaic filler that machine-generated or engagement-farm copy leans on.
SLOP_CLICHES = {
    "in conclusion", "moreover", "furthermore", "it is important to note",
    "worth noting", "delve", "delving", "tapestry", "landscape of",
    "fast-paced world", "as we navigate", "navigate the complexities",
    "underscores", "underscored", "highlights the importance",
    "raises questions", "sparks debate", "sheds light on", "myriad",
    "plethora", "realm of", "ever-evolving", "game-changer",
    "revolutionize", "unprecedented times", "in the realm",
    "it remains to be seen", "only time will tell", "fingers pointed",
    "eyes on", "watch this space", "development is unfolding",
    "more details to follow", "stay tuned",
}

# Attribution that names nobody - the classic slop tell.
VAGUE_ATTRIBUTION = {
    "experts say", "sources say", "reportedly", "allegedly", "rumoured",
    "rumored", "many believe", "critics say", "observers say", "some claim",
    "some say", "unnamed sources", "insiders reveal", "people are saying",
    "it is believed", "is believed to", "speculated", "suggests that many",
}

NAMED_SOURCE_RE = re.compile(
    r"\b(?:according to|said|tells?|tweeted|posted)\s+[A-Z]\w+(?:\s+[A-Z]\w+){0,2}\b")
CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
ACRONYM_OK = {
    "CERT", "CBI", "FBI", "CIA", "NDTV", "BBC", "CNN", "ESPN", "NATO",
    "NASA", "ISRO", "WHO", "IMF", "WTO", "OPEC", "NCAA", "NBA", "NFL",
    "USA", "UK", "UAE", "EU", "UN", "DGP", "SSC", "RBI", "SEBI", "SUPREME",
    "GSI", "RAW", "CRPF", "BSF", "CISF", "NSG", "NDRF", "SDRF", "BMD",
}

FIRST_PERSON_OPINION = re.compile(
    r"\b(i think|i believe|we must|we need to|we should|our fight|i argue|"
    r"in my view|in my opinion|let us|let's)\b", re.IGNORECASE)
MODAL_OBLIGATION = re.compile(r"\b(must|should|ought to|cannot be allowed)\b", re.IGNORECASE)
SUPERLATIVE = re.compile(r"\b(biggest|worst|greatest|best|deadliest|unprecedented|"
                         r"unparalleled|never before)\b", re.IGNORECASE)
QUESTION = re.compile(r"^[^?]+\?", )
NUMERIC = re.compile(r"\b\d+([.,]\d+)?%?\b|\$\d|\b\d{4}\b")
CLICKBAIT = re.compile(r"\b(you won't believe|this is why|here's (what|why)|"
                       r"shocking truth|what happens next|because of this|"
                       r"never be the same|destroyed everything)\b", re.IGNORECASE)

NEG_SENTENCE = -0.45   # VADER compound at/under this marks a negative sentence
POS_SENTENCE = 0.45

SOURCING_BONUS = 18.0  # informational head start: most news *intent* is to inform


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter tuned for news copy.

    Paragraph boundaries are hard breaks: news HTML often drops terminal
    punctuation, so sentences from different paragraphs would otherwise fuse.
    """
    out: list[str] = []
    for paragraph in PARA_SPLIT.split(text.replace("\r", "")):
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if not paragraph:
            continue
        # keep abbreviations from splitting: "Mr. Smith", "U.S. Navy"
        paragraph = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|no)\.\s", r"\1<DOT> ", paragraph)
        paragraph = re.sub(r"\b(U|US|UK|UAE|EU|P|H)\.\s", r"\1<DOT> ", paragraph)
        for sentence in SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.replace("<DOT>", ".").strip()
            if len(sentence) > 20:
                out.append(sentence)
    return out

_TERM_CACHE: dict[frozenset[str], list[re.Pattern]] = {}


def _term_patterns(terms: set[str]) -> list[re.Pattern]:
    """Compile lexicon terms into whole-word regex patterns.

    Stems ("injur", "devastat") are extended with common English suffixes;
    full words match only themselves, so "toward" never trips "war" and
    "deadline" never trips "dead".
    """
    key = frozenset(terms)
    compiled = _TERM_CACHE.get(key)
    if compiled is None:
        suffix = r"(?:s|es|ed|d|ing|ion|ions|y|e)?"
        compiled = [
            re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            if " " in term
            else re.compile(r"\b" + re.escape(term) + suffix + r"\b", re.IGNORECASE)
            for term in sorted(terms)
        ]
        _TERM_CACHE[key] = compiled
    return compiled


def _count_terms(text: str, terms: set[str]) -> int:
    return sum(len(pattern.findall(text)) for pattern in _term_patterns(terms))


def _saturate(raw: float) -> float:
    """Map an unbounded raw strength to a 0-100 scale that grows but never pegs."""
    return 100.0 * (1.0 - math.exp(-raw))


def reading_ease(text: str) -> float:
    """Flesch Reading Ease with a syllable heuristic. 0 = impenetrable, 100 = easy."""
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 10:
        return 0.0
    syllables = 0
    for w in words:
        groups = re.findall(r"[aeiouy]+", w.lower())
        count = len(groups)
        if w.lower().endswith("e") and count > 1:
            count -= 1
        syllables += max(count, 1)
    sentences = max(len(split_sentences(text)), 1)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return round(max(0.0, min(100.0, score)), 1)


def analyze(text: str) -> dict:
    """Run the full pipeline over cleaned article text."""
    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return {"error": "Not enough article text was extracted to analyse."}

    overall = _VADER.polarity_scores(text)
    words = len(text.split())
    neg_sents = pos_sents = 0
    harm_hits = statement_hits = source_hits = 0
    numeric_sents = opinion_sents = question_sents = clickbait_sents = 0
    quote_sents = 0
    slop_hits = vague_hits = named_hits = 0
    superlative_sents = 0
    harm_evidence: list[tuple[float, str]] = []
    agenda_evidence: list[tuple[float, str]] = []
    slop_evidence: list[tuple[float, str]] = []

    for sentence in sentences:
        scores = _VADER.polarity_scores(sentence)
        compound = scores["compound"]
        if compound <= NEG_SENTENCE:
            neg_sents += 1
        if compound >= POS_SENTENCE:
            pos_sents += 1

        s_low = sentence.lower()
        h = _count_terms(sentence, HARM_TERMS)
        a = _count_terms(sentence, AGENDA_TERMS)
        src = _count_terms(sentence, SOURCE_TERMS)
        s = _count_terms(s_low, SLOP_CLICHES)
        v = _count_terms(s_low, VAGUE_ATTRIBUTION)

        if s or v:
            slop_hits += s
            vague_hits += v
            slop_evidence.append((s + 2.0 * v, sentence))

        if h:
            harm_hits += h
            weight = h + (1.0 if compound <= NEG_SENTENCE else 0.0)
            if compound <= NEG_SENTENCE or h >= 2:
                harm_evidence.append((weight, sentence))

        opinion_local = 0
        sourced = src > 0
        if FIRST_PERSON_OPINION.search(sentence):
            opinion_local += 2
        # "must possess a certificate" inside reported policy is sourcing, not sermonising
        if MODAL_OBLIGATION.search(s_low):
            opinion_local += 0.4 if sourced else 1.0
        if SUPERLATIVE.search(s_low):
            opinion_local += 0.4 if sourced else 1.0
        opinion_local += a
        if opinion_local:
            statement_hits += 1 + (opinion_local - 1) * 0.5
            agenda_evidence.append((opinion_local + (1.0 if compound <= NEG_SENTENCE else 0.0), sentence))

        if src:
            source_hits += 1
        named_hits += len(NAMED_SOURCE_RE.findall(sentence))
        if SUPERLATIVE.search(s_low):
            superlative_sents += 1
        if NUMERIC.search(sentence):
            numeric_sents += 1
        if '"' in sentence or "\u201c" in sentence:
            quote_sents += 1
        if "?" in sentence and sentence.strip().endswith("?"):
            question_sents += 1
        if CLICKBAIT.search(s_low):
            clickbait_sents += 1
            agenda_evidence.append((2.0, sentence))

    # ------------------------------------------------ scoring --------------
    neg_density = neg_sents / n
    harm_density = harm_hits / n
    exclaim = text.count("!") / max(text.count("."), 1)
    opinion_density = statement_hits / n

    harmful_score = _saturate(2.6 * neg_density + 1.9 * harm_density + 0.5 * min(exclaim, 0.25))

    # Editorial precedence: when persuasion markers dominate the article,
    # its purpose is advocacy, so Harmful must yield ground to Agenda-Based.
    harmful_score *= 1.0 - min(opinion_density * 1.2, 0.55)

    clickbait_density = clickbait_sents / n
    question_density = question_sents / n
    agenda_score = _saturate(3.2 * opinion_density + 2.5 * clickbait_density + 0.6 * question_density)

    source_density = source_hits / n
    quote_density = quote_sents / n
    numeric_density = numeric_sents / n
    neutrality = 1.0 - min(1.0, abs(overall["compound"]) * 1.6)
    info_raw = SOURCING_BONUS + 28.0 * source_density + 14.0 * quote_density + \
        16.0 * numeric_density + 14.0 * neutrality - 22.0 * opinion_density - 8.0 * neg_density
    informational_score = max(0.0, min(100.0, info_raw))

    # ―――――――――――――― slop vs real meter ――――――――――――――
    # Signal 1: formulaic filler the machine-generated & engagement-farm
    #           copy leans on ("moreover", "delve", "sparks debate").
    # Signal 2: attributions that name nobody ("experts say").
    # Signal 3: burstiness - different sentence lengths is an edited-writing
    #           tell; uniform rhythm is the machine tell.
    # Signal 4: shouting (ALL-CAPS words) and superlative density.
    # Signal 5 (real-side): specificity - named sources, quotes, numbers.
    lengths = [len(re.findall(r"\w+", sentence)) for sentence in sentences]
    mean_len = sum(lengths) / max(n, 1)
    variance = sum((L - mean_len) ** 2 for L in lengths) / max(n, 1)
    burstiness = (variance ** 0.5) / max(mean_len, 1.0)
    uniformity = 1.0 - min(burstiness, 0.9) / 0.9   # 0 = varied, 1 = metronome

    lower_words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    shout_count = len(
        [word for word in CAPS_RE.findall(text)
         if word not in ACRONYM_OK and word.lower() in lower_words])
    shout_per_kword = shout_count / max(words, 1) * 1000.0

    slop_density = slop_hits / n
    vague_density = vague_hits / n
    superlative_density = superlative_sents / n
    # shouting only counts at tabloid concentrations: acronym-dense legal and
    # tech reporting uses ALL-CAPS constantly without being slop
    shout_excess = max(0.0, shout_per_kword - 15.0)
    slop_raw = (
        1.2 * slop_density + 1.0 * vague_density + 1.2 * uniformity
        + min(shout_excess, 20.0) * 0.08 + 0.8 * superlative_density
        - 0.15 * min(source_hits, 8))
    # logistic: ~0 raw -> near 0, 3.0 -> coin-flip, 5+ -> saturated slop
    slop_score = 100.0 / (1.0 + math.exp(-1.2 * (slop_raw - 3.0)))

    named_density = named_hits / n
    real_raw = 30.0 + 26.0 * source_density + 22.0 * named_density
    real_raw += 14.0 * quote_density + 12.0 * numeric_density
    real_raw += 10.0 * (1.0 - uniformity)
    real_raw -= 10.0 * min(4.0 * slop_density, 1.0)
    real_score = max(0.0, min(100.0, real_raw))

    ranked = sorted(
        [("Informational", informational_score),
         ("Harmful", harmful_score),
         ("Agenda-Based", agenda_score)],
        key=lambda item: item[1], reverse=True)
    verdict, top = ranked[0]
    margin = top - ranked[1][1]

    def top_evidence(items: list[tuple[float, str]], limit: int = 4) -> list[str]:
        return [s for _, s in sorted(items, key=lambda x: x[0], reverse=True)[:limit]]

    minutes = max(1, round(words / 220))
    return {
        "verdict": verdict,
        "confidence": round(min(margin, 60.0) / 60.0 * 100.0, 1),
        "scores": {
            "Informational": round(informational_score, 1),
            "Harmful": round(harmful_score, 1),
            "Agenda-Based": round(agenda_score, 1),
        },
        "sentiment": {
            "positive": round(overall["pos"] * 100, 1),
            "neutral": round(overall["neu"] * 100, 1),
            "negative": round(overall["neg"] * 100, 1),
            "compound": round(overall["compound"], 3),
        },
        "evidence": {
            "Harmful": top_evidence(harm_evidence),
            "Agenda-Based": top_evidence(agenda_evidence),
            "Informational": _source_evidence(sentences),
        },
        "stats": {
            "sentences": n,
            "negative_sentences": neg_sents,
            "opinion_sentences": round(statement_hits, 1),
            "sourced_sentences": source_hits,
            "reading_ease": reading_ease(text),
            "reading_time_min": minutes,
        },
        "slop": {
            "slop_score": round(slop_score, 1),
            "real_score": round(real_score, 1),
            "signals": {
                "slop_phrases": slop_hits,
                "vague_attributions": vague_hits,
                "named_sources": named_hits,
                "shouting_words": shout_count,
                "sentence_uniformity_pct": round(uniformity * 100, 1),
            },
            "evidence": top_evidence(slop_evidence),
        },
    }


def _source_evidence(sentences: list[str]) -> list[str]:
    """Sentences that best demonstrate sourced, factual reporting."""
    scored = []
    for sentence in sentences:
        s = _count_terms(sentence, SOURCE_TERMS)
        if not s:
            continue
        score = s + len(NUMERIC.findall(sentence)) * 0.5
        if '"' in sentence or '\u201c' in sentence:
            score += 1
        scored.append((score, sentence))
    return [x for _, x in sorted(scored, key=lambda t: t[0], reverse=True)[:4]]
