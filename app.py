"""
app.py -- PressLens: Sentiment Analysis of Live News Using Web Scraping.

Paste a live news article link. The app scrapes the article, analyses its
sentiment and classifies its purpose: Informational, Harmful, or Agenda-Based.

Run:  streamlit run app.py
"""

import datetime
import re

import streamlit as st

from analyzer import analyze
from scraper import extract_article, fetch, suggest_article_links

st.set_page_config(
    page_title="PressLens - Live News Sentiment Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ theme ---
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Libre+Caslon+Text:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

      :root {
        --paper: #faf8f2;
        --ink:   #17150f;
        --rule:  #d8d2c4;
        --faint: #8d8577;
        --crimson: #a11d21;
        --amber:   #94610b;
        --steel:   #1d3f6e;
      }

      .stApp {
        background: var(--paper);
        color: var(--ink);
      }
      .stApp, .stMarkdown, .stText { font-family: 'Libre Caslon Text', Georgia, serif; }

      #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
      .block-container { max-width: 1080px; padding-top: 2.2rem; }

      a { color: var(--steel); }

      /* masthead ---------------------------------------------------------- */
      .masthead-wrap {
        border-top: 3px solid var(--ink);
        border-bottom: 1px solid var(--ink);
        padding: 10px 0 12px 0;
        margin-bottom: 6px;
      }
      h1.masthead {
        font-family: 'Libre Caslon Text', Georgia, serif;
        font-size: 3.0rem; font-weight: 700; letter-spacing: -0.02em;
        margin: 0; line-height: 1.0; color: var(--ink);
      }
      .masthead-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem; letter-spacing: 0.22em; text-transform: uppercase;
        color: var(--faint); margin-top: 6px;
      }
      .dateline {
        display: flex; justify-content: space-between;
        border-bottom: 1px solid var(--rule); padding-bottom: 8px; margin-bottom: 28px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--faint);
      }

      /* input row --------------------------------------------------------- */
      .stTextInput > div, .stTextInput [data-baseweb="input"] {
        background: var(--paper) !important; border: none !important; box-shadow: none !important;
      }
      .stTextInput input {
        border: none; border-bottom: 2px solid var(--ink); border-radius: 0;
        background: var(--paper) !important; color: var(--ink) !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem; padding: 10px 2px;
      }
      .stTextInput input:focus { box-shadow: none; border-bottom-color: var(--crimson); }
      .stButton > button {
        border: 1.5px solid var(--ink); border-radius: 0; background: transparent;
        color: var(--ink); font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.18em; text-transform: uppercase; font-size: 0.72rem;
        padding: 10px 22px; transition: all .15s ease;
      }
      .stButton > button:hover { background: var(--ink); color: var(--paper); border-color: var(--ink); }
      .stButton > button[kind="secondary"] { border-color: var(--rule); color: var(--ink); }

      /* notices ----------------------------------------------------------- */
      .notice { border: 1.5px solid; padding: 18px 24px 16px 24px; margin: 6px 0 18px 0; }
      .notice-title {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
        letter-spacing: 0.24em; text-transform: uppercase;
      }
      .notice-body { font-size: 0.98rem; margin-top: 8px; line-height: 1.55; }

      /* verdict ----------------------------------------------------------- */
      .verdict { border: 1.5px solid var(--ink); padding: 26px 30px 22px 30px; margin: 8px 0 26px 0; }
      .verdict-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
        letter-spacing: 0.26em; text-transform: uppercase; color: var(--faint);
      }
      .verdict-word {
        font-family: 'Libre Caslon Text', Georgia, serif;
        font-size: 2.6rem; font-weight: 700; line-height: 1.05; margin-top: 4px;
      }
      .verdict-note { font-size: 0.92rem; color: var(--faint); margin-top: 10px; font-style: italic; }
      .v-info   { color: var(--steel); }
      .v-harm   { color: var(--crimson); }
      .v-agenda { color: var(--amber); }

      /* score bars -------------------------------------------------------- */
      .score-row { margin-bottom: 16px; }
      .score-head {
        display: flex; justify-content: space-between; align-items: baseline;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 5px;
      }
      .score-track { height: 8px; background: #e9e4d8; position: relative; }
      .score-fill { position: absolute; top: 0; left: 0; bottom: 0; }

      /* evidence ---------------------------------------------------------- */
      .section-rule {
        border-top: 1px solid var(--ink); margin: 26px 0 14px 0; padding-top: 10px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
        letter-spacing: 0.24em; text-transform: uppercase; color: var(--ink);
      }
      .evidence-block {
        border-left: 3px solid var(--rule); margin: 12px 0 16px 0; padding: 4px 0 4px 16px;
        font-size: 0.98rem; line-height: 1.55;
      }
      .evidence-block.harm { border-left-color: var(--crimson); }
      .evidence-block.agenda { border-left-color: var(--amber); }
      .evidence-block.info { border-left-color: var(--steel); }
      .evidence-tag {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem;
        letter-spacing: 0.2em; text-transform: uppercase; color: var(--faint);
        margin-bottom: 4px;
      }

      /* metrics ----------------------------------------------------------- */
      [data-testid="stMetric"] {
        border-top: 1px solid var(--rule); padding-top: 10px;
      }
      [data-testid="stMetricLabel"] p {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.66rem !important; letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--faint) !important;
      }
      [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
      }
      .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        letter-spacing: 0.16em; text-transform: uppercase;
      }
      .stTabs [aria-selected="true"] { border-bottom: 2px solid var(--ink) !important; }

      .footnote {
        border-top: 1px solid var(--rule); margin-top: 40px; padding-top: 10px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
        color: var(--faint); letter-spacing: 0.1em;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

VERDICT_CLASS = {"Informational": "v-info", "Harmful": "v-harm", "Agenda-Based": "v-agenda"}
VERDICT_BLAZON = {
    "Informational": ("var(--steel)", "Informational"),
    "Harmful": ("var(--crimson)", "Harmful"),
    "Agenda-Based": ("var(--amber)", "Agenda-Based"),
}
VERDICT_NOTES = {
    "Informational": "The article reads primarily as sourced, factual reporting.",
    "Harmful": "The article carries heavily negative or distressing framing.",
    "Agenda-Based": "The article works to persuade rather than simply inform.",
}

EXAMPLE_URL = ("https://www.thehindu.com/news/national/"
               "road-ministry-seeks-forensic-audit-of-its-portals-"
               "after-vehicle-data-leak/article71620455.ece")

NOTICE_COLOR = {"warn": "var(--amber)", "error": "var(--crimson)", "info": "var(--steel)"}


def dateline() -> str:
    return datetime.date.today().strftime("%A, %d %B %Y")


def notice(kind: str, title: str, body: str) -> None:
    """An in-theme paper instead of Streamlit's default alert blobs."""
    color = NOTICE_COLOR[kind]
    st.markdown(
        f"""
        <div class="notice" style="border-color:{color};">
          <div class="notice-title" style="color:{color};">{title}</div>
          <div class="notice-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_bar(label: str, value: float, color: str, dominant: bool) -> str:
    flag = " - dominant" if dominant else ""
    return f"""
    <div class="score-row">
      <div class="score-head"><span>{label}{flag}</span><span>{value:.1f} / 100</span></div>
      <div class="score-track"><div class="score-fill" style="width:{value:.1f}%;background:{color};"></div></div>
    </div>"""


def pick_article(link: str) -> None:
    st.session_state["picked_url"] = link


def run_report(target: str, depth: int = 0) -> None:
    try:
        with st.spinner("Scraping the article..."):
            html = fetch(target)
    except Exception as exc:
        notice("error", "Scrape failed",
               f"Could not reach the page: <code style='font-family:IBM Plex Mono,monospace;'>{exc}</code>")
        return

    article = extract_article(target, html)

    if article["word_count"] < 80:
        notice(
            "warn",
            "No readable article on this link",
            "This looks like a homepage, a section page, or a heavily "
            "JavaScript-driven site - PressLens found no article body here. "
            "Paste the <b>direct link to a single article</b> instead "
            "(open the news site and click one headline).",
        )
        if depth == 0:
            suggestions = suggest_article_links(target, html)
            if suggestions:
                st.markdown(
                    '<div class="section-rule">Articles found on this page - one click to analyse</div>',
                    unsafe_allow_html=True,
                )
                for i, (link, headline) in enumerate(suggestions):
                    st.button(headline[:110], key=f"sug_{i}",
                              on_click=pick_article, args=(link,),
                              use_container_width=True)
        return

    result = analyze(article["text"])
    if result.get("error"):
        notice("error", "Analysis failed", result["error"])
        return

    # ------------------------------------------------------------- verdict --
    color, word = VERDICT_BLAZON[result["verdict"]]
    st.markdown(
        f"""
        <div class="verdict">
          <div class="verdict-label">Verdict &middot; {article['domain']}</div>
          <div class="verdict-word {VERDICT_CLASS[result['verdict']]}">{word}</div>
          <div class="verdict-note">{VERDICT_NOTES[result['verdict']]}
            &nbsp;&mdash;&nbsp; confidence {result['confidence']:.0f}% (margin over next class)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------- article identity + purpose score bars --
    col_title, col_meta = st.columns([3, 1.4])
    with col_title:
        st.markdown(f"### {article['title'] or '(untitled)'}")
        st.caption(f"{article['domain']} - {article['word_count']} words scraped")
    with col_meta:
        st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;line-height:2.0;color:var(--faint);letter-spacing:0.1em;">
              READING EASE &nbsp;{result['stats']['reading_ease']:.0f}<br>
              READ TIME &nbsp;&nbsp;&nbsp;~{result['stats']['reading_time_min']} MIN<br>
              SENTENCES &nbsp;&nbsp;&nbsp;{result['stats']['sentences']}<br>
              SOURCED &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{result['stats']['sourced_sentences']}
            </div>
        """, unsafe_allow_html=True)

    s = result["scores"]
    st.markdown('<div class="section-rule">Purpose Scores</div>', unsafe_allow_html=True)
    st.markdown(
        score_bar("Informational", s["Informational"], "var(--steel)", result["verdict"] == "Informational")
        + score_bar("Harmful", s["Harmful"], "var(--crimson)", result["verdict"] == "Harmful")
        + score_bar("Agenda-Based", s["Agenda-Based"], "var(--amber)", result["verdict"] == "Agenda-Based"),
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------ slop meter --
    slop = result["slop"]
    slop_val, real_val = slop["slop_score"], slop["real_score"]
    if slop_val >= 55:
        meter_label = "This reads like slop - formulaic, unnamed, engagement-bait styling"
        meter_color, meter_class = "var(--crimson)", "v-harm"
    elif slop_val >= 35:
        meter_label = "Mixed signals - real bones, but lazy machine-flavoured filler throughout"
        meter_color, meter_class = "var(--amber)", "v-agenda"
    else:
        meter_label = "This reads like real reporting - sourced, specific, edited"
        meter_color, meter_class = "var(--steel)", "v-info"

    st.markdown('<div class="section-rule">Slop / Real Meter</div>', unsafe_allow_html=True)
    st.markdown(
        score_bar("Real news", real_val, "var(--steel)", real_val > slop_val)
        + score_bar("Slop", slop_val, "var(--crimson)", slop_val >= real_val),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="border-left:3px solid {meter_color}; padding:2px 0 2px 14px; margin:2px 0 16px 0;">
          <div class="verdict-word {meter_class}" style="font-size:1.35rem;">{meter_label}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:0.1em;color:var(--faint);margin-top:8px;">
            FORMULAIC FILLER {slop['signals']['slop_phrases']} &middot; UNNAMED ATTRIBUTION {slop['signals']['vague_attributions']}
            &middot; NAMED SOURCES {slop['signals']['named_sources']} &middot; SHOUTING {slop['signals']['shouting_words']}
            &middot; SENTENCE-RHYTHM UNIFORMITY {slop['signals']['sentence_uniformity_pct']:.0f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if slop["evidence"]:
        st.markdown('<div class="evidence-tag">Sloppiest sentences</div>', unsafe_allow_html=True)
        for line in slop["evidence"][:3]:
            st.markdown(
                f'<div class="evidence-block harm" style="margin:8px 0;">'
                f'<div style="margin-bottom:0;">&ldquo;{line}&rdquo;</div></div>',
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------- evidence tabs --
    st.markdown('<div class="section-rule">Evidence &amp; Sentiment</div>', unsafe_allow_html=True)
    tab_evidence, tab_sentiment, tab_text = st.tabs(["Evidence", "Sentiment", "Scraped Text"])

    with tab_evidence:
        for klass, label in [("info", "Informational cues"), ("harm", "Harmful cues"), ("agenda", "Agenda cues")]:
            key = {"info": "Informational", "harm": "Harmful", "agenda": "Agenda-Based"}[klass]
            lines = result["evidence"][key]
            if not lines:
                st.markdown(
                    f'<div class="evidence-tag">{label} &mdash; none found</div>'
                    f'<p style="font-size:0.9rem;color:var(--faint);font-style:italic;">No supporting sentences.</p>',
                    unsafe_allow_html=True,
                )
                continue
            blocks = ""
            for line in lines:
                blocks += f'<div style="margin-bottom:8px;">&ldquo;{line}&rdquo;</div>'
            st.markdown(
                f'<div class="evidence-tag">{label}</div>'
                f'<div class="evidence-block {klass}">{blocks}</div>',
                unsafe_allow_html=True,
            )

    with tab_sentiment:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Positive", f"{result['sentiment']['positive']:.1f}%")
        m2.metric("Neutral", f"{result['sentiment']['neutral']:.1f}%")
        m3.metric("Negative", f"{result['sentiment']['negative']:.1f}%")
        compound = result["sentiment"]["compound"]
        m4.metric(
            "Compound",
            f"{compound:+.3f}",
            delta="negative-leaning" if compound < -0.05 else
                  ("positive-leaning" if compound > 0.05 else "neutral"),
            delta_color="inverse" if compound < -0.05 else "normal",
        )
        st.caption(
            "Compound runs -1 (most negative) to +1 (most positive), computed by "
            "VADER across the full article."
        )
        st.caption(
            f"Article tone: {result['stats']['negative_sentences']} of "
            f"{result['stats']['sentences']} sentences are strongly negative; "
            f"{result['stats']['opinion_sentences']:.1f} carry opinion/persuasion markers."
        )

    with tab_text:
        paragraphs = article["paragraphs"]
        st.caption(f"{len(paragraphs)} scraped paragraphs")
        for para in paragraphs:
            st.write(para)

    st.markdown(
        f'<div class="footnote">PRESSLENS &middot; SCRAPER: REQUESTS + BEAUTIFULSOUP '
        f'&middot; SENTIMENT: VADER &middot; CLASSIFIER: LEXICON + DENSITY HEURISTIC &middot; '
        f'SOURCE: {article["domain"].upper()}</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------- flow ---

st.markdown(
    f"""
    <div class="masthead-wrap">
      <h1 class="masthead">PressLens</h1>
      <div class="masthead-sub">Sentiment Analysis of Live News &middot; Web Scraping &middot; Purpose Classification</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""<div class="dateline"><span>Vol. I &middot; Mini Project</span><span>{dateline()}</span></div>""",
    unsafe_allow_html=True,
)

st.markdown(
    "Paste the link of any live news article. The article is scraped, cleaned, "
    "analysed for sentiment, and classified by purpose.",
    unsafe_allow_html=True,
)

picked = st.session_state.pop("picked_url", None)

left, right = st.columns([4.2, 1.15])
with left:
    url = st.text_input("Article URL", placeholder="https://...", label_visibility="collapsed")
with right:
    go = st.button("Analyse", use_container_width=True)

st.markdown(
    "<span style='font-family:IBM Plex Mono,monospace;font-size:0.66rem;letter-spacing:0.12em;color:var(--faint);'>"
    "TIP: PASTE A SINGLE ARTICLE LINK - NOT A HOMEPAGE.</span>",
    unsafe_allow_html=True,
)
st.button("Load an example article", key="example_btn",
          on_click=pick_article, args=(EXAMPLE_URL,))

target = None
if go and url.strip():
    if not re.match(r"^https?://", url.strip()):
        notice("error", "Not a link",
               "An article URL starts with <code>http://</code> or <code>https://</code>.")
    else:
        target = url.strip()
elif picked:
    target = picked
elif go:
    notice("info", "Waiting on a link", "Paste an article URL above, then press Analyse.")

if target:
    run_report(target)
