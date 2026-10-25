# PressLens - Sentiment Analysis of Live News Using Web Scraping

Paste a live news article link. The article is scraped, cleaned, analysed for
sentiment, and classified by purpose: **Informational**, **Harmful**, or
**Agenda-Based**, with a report showing the evidence.

Built as the mini project *Sentiment Analysis of Live News Using Web Scraping*
(Abdullah Shaikh, TYCS-B, Roll No: 6076), per the approved synopsis.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and paste any article URL.

## How it works

1. **Scrape** (`scraper.py`) - Requests + BeautifulSoup fetch the page with a
   browser User-Agent, strip site chrome, pick the prose-heaviest paragraph
   container, and clean boilerplate, URLs, and duplicate lead lines.
2. **Analyse** (`analyzer.py`) - VADER scores sentiment per sentence and for the
   whole article. Three purpose scores are computed from evidence-weighted
   lexicons:
   - *Informational*: sourcing density ("according to", officials, data,
     numbers, quotes, dates) and neutrality.
   - *Harmful*: negative-sentence density plus violence/disaster lexicon.
   - *Agenda-Based*: persuasion markers (first-person advocacy, obligation
     modals outside reported speech, superlatives, emotive/conspiracy terms,
     clickbait), with editorial precedence over Harmful when opinion density
     dominates.
3. **Report** (`app.py`) - verdict with confidence margin, purpose score bars,
   evidence sentences for each class, a **Slop / Real meter** (how much the
   article reads like machine-generated or engagement-bait copy vs. edited
   sourced reporting), VADER sentiment metrics, reading stats, and the scraped
   text.

The Slop meter weighs five signals: formulaic filler phrases ("moreover",
"delve into", "sparks debate"), attributions that name nobody ("experts say"),
sentence-rhythm uniformity (low burstiness), ALL-CAPS shouting in excess, and
hyperbolic superlatives - against the real-side signals of named sources,
quotes, numbers and sourcing. Works offline, no AI APIs.

No paid AI services or API keys - everything runs locally.

## Files

- `app.py` - Streamlit report UI (editorial newsprint theme)
- `scraper.py` - fetch + extract + clean
- `analyzer.py` - sentiment + purpose classification + slop detection
- `dev_probe.py` - regression checks for the slop meter (`python3 dev_probe.py`)
- `requirements.txt` - streamlit, requests, beautifulsoup4, vaderSentiment
