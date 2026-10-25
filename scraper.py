"""
scraper.py -- Fetches a live news page and extracts the article title and body.

Strategy:
  1. Download the HTML with a real browser User-Agent.
  2. Strip scripts, styles, nav, headers, footers, ads, forms.
  3. Group <p> tags under their parent container and pick the container
     with the most substantial prose (the article body).
  4. Clean every paragraph: drop boilerplate, nav junk, URLs, duplicates.
"""

import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Lines that are site chrome, not journalism.
BOILER_PATTERNS = re.compile(
    r"^(sign in|sign up|log in|subscribe|register|advertisement|sponsored|"
    r"share (this|on)|follow us|cookie|privacy|terms of use|all rights reserved|"
    r"read more|click here|continue reading|advertisement|trending now|"
    r"related (articles|stories|news)|most read|top stories|newsletter|"
    r"download now|download the app|listen to (this )?article|skip to (main )?content)\b",
    re.IGNORECASE,
)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
PROMO_RE = re.compile(
    r"\b(subscribe|newsletter|podcast|epaper|e-paper|download|sign (in|up)|register|see all)\b",
    re.IGNORECASE)
FT_RE = re.compile(r"\b(?:[a-z]{1,2}\.?[- ]?){1,5}FT\b|\(FT\)|\bgetty\b|reuters/?", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def fetch(url: str, timeout: int = 20) -> str:
    """Download the page HTML. Raises requests exceptions on failure."""
    with requests.Session() as session:
        session.headers.update(HEADERS)
        response = session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", "text/html"):
            raise ValueError("The URL did not return an HTML page.")
        return response.text


def _clean_line(line: str) -> str:
    """Normalise one paragraph of raw page text."""
    line = URL_RE.sub("", line)
    line = FT_RE.sub("", line)
    line = WHITESPACE_RE.sub(" ", line).strip()
    # leading editorial junk like "READ MORE:" or "- Advertisement -"
    line = re.sub(r"^[\W_]+|[\W_]+$", "", line)
    return line


def _is_junk(line: str) -> bool:
    if len(line) < 40:
        return True
    if BOILER_PATTERNS.match(line):
        return True
    letters = sum(c.isalpha() for c in line)
    if letters / max(len(line), 1) < 0.55:  # mostly symbols/numbers
        return True
    return False


def _pick_title(soup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return WHITESPACE_RE.sub(" ", og["content"]).strip()
    if soup.title and soup.title.string:
        return WHITESPACE_RE.sub(" ", soup.title.string).strip()
    if soup.h1:
        return soup.h1.get_text(" ", strip=True)
    return ""


def _extract_body(soup) -> list[str]:
    """Group all <p> tags by parent and score each group, keeping the winner."""
    groups: dict[object, list[str]] = {}
    for p in soup.find_all("p"):
        text = _clean_line(p.get_text(" ", strip=True))
        if _is_junk(text):
            continue
        # walk up to a container that holds prose (div, article, section...)
        node = p.parent
        while node is not None and node.name not in ("div", "article", "section", "main", "body"):
            node = node.parent
        key = node if node is not None else p.parent
        groups.setdefault(key, []).append(text)

    best, best_score = [], 0
    for paras in groups.values():
        total = sum(len(t) for t in paras)
        score = total * (1 + min(len(paras) // 4, 3) * 0.25)
        if score > best_score:
            best_score, best = score, paras

    # de-duplicate near-identical paragraphs (news sites repeat lead lines)
    seen: set[str] = set()
    unique = []
    for t in best:
        fingerprint = t[:80].lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(t)
    return unique


def suggest_article_links(url: str, html: str, limit: int = 6) -> list[tuple[str, str]]:
    """For section/homepage URLs: find deep, headline-bearing same-domain links.

    A user pasting a front page still gets a useful next step: the readable
    articles hiding on it.
    """
    base_domain = urlparse(url).netloc.removeprefix("www.")
    soup = BeautifulSoup(html, "html.parser")
    suggestions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        link = urljoin(url, anchor["href"].split("#")[0].split("?")[0])
        parsed = urlparse(link)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc.removeprefix("www.") != base_domain:
            continue
        segments = [s for s in parsed.path.split("/") if s]
        if len(segments) < 2:  # shallow = another section page
            continue
        headline = WHITESPACE_RE.sub(" ", anchor.get_text(" ", strip=True)).strip()
        # headlines are short; promo blurbs are long, salesy, or newsletter bait
        if len(headline) < 30 or len(headline.split()) > 14:
            continue
        if PROMO_RE.search(headline) or link in seen:
            continue
        seen.add(link)
        suggestions.append((link, headline))
        if len(suggestions) >= limit:
            break
    return suggestions


def extract_article(url: str, html: str) -> dict:
    """Build the article record: title, clean text, domain, stats."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "aside", "form", "iframe", "figure", "button", "svg"]):
        tag.decompose()

    title = _pick_title(soup)
    paragraphs = _extract_body(soup)
    text = "\n\n".join(paragraphs)
    words = len(text.split())

    domain = urlparse(url).netloc.removeprefix("www.")

    return {
        "url": url,
        "domain": domain,
        "title": title,
        "text": text,
        "paragraphs": paragraphs,
        "word_count": words,
    }
