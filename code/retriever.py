import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from config import CORPUS_PATHS, BM25_TOP_K, ROOT

# Trap files that should never be retrieved (deprecated, out-of-scope, or generic
# index files). Excluded from every area's index at build time.
EXCLUDED_FILES = {
    "api-reference-deprecated-endpoints.md",
    "understanding-large-language-models-primer.md",
    "hackerrank-subscription-management.md",
    "changelog-visa-policy-updates-q1-2026.md",
    "comprehensive-history-digital-payment-networks.md",
    "index.md",
    "support.md",  # top-level generic index file — excluded at root level only
}

# Small English stopword set. Removed from BOTH the indexed text and the query
# so common words (how/do/i/my) don't dominate BM25 scoring.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "of", "to",
    "in", "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "am", "do", "does", "did", "done", "doing", "have", "has", "had",
    "having", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that", "these",
    "those", "what", "which", "who", "whom", "whose", "how", "when", "where", "why",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must", "not",
    "no", "so", "than", "too", "very", "just", "about", "into", "over", "out",
    "please", "there", "here",
}

# Chunking parameters (word counts).
MIN_WORDS = 60        # below this a section is merged with neighbours
TARGET_WORDS = 220    # accumulate sections up to this; window size when splitting
MAX_WORDS = 320       # sections above this are window-split (fits embed/BM25 well)
OVERLAP_WORDS = 30    # carried between windows so facts aren't lost at the seam

# Relevance gate + diversity.
RELATIVE_FLOOR = 0.25      # drop hits scoring below this fraction of the top hit
MAX_CHUNKS_PER_DOC = 2     # don't let one file fill the whole result set

INDEXES = {}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords. Used for both the
    index and the query so matching stays symmetric (and punctuation-proof)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter, body). Only treats a leading, newline-delimited
    --- ... --- block as frontmatter, so markdown thematic breaks are safe."""
    m = re.match(r"---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if m:
        return m.group(1), text[m.end():]
    return "", text


def _extract_title(frontmatter: str, body: str) -> str:
    m = re.search(r'^title:\s*(.+?)\s*$', frontmatter, re.MULTILINE)
    if m:
        return m.group(1).strip().strip('"\'')
    h = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    return h.group(1).strip() if h else ""


def _extract_breadcrumbs(frontmatter: str) -> list[str]:
    return [c.strip().strip('"\'') for c in re.findall(r'^\s*-\s*(.+?)\s*$', frontmatter, re.MULTILINE)]


def clean_content(text: str) -> str:
    """Strip non-content noise from a markdown body: Related Articles blocks,
    link URLs (keep anchor text), bare URLs, the _Last updated_ line, and
    excess blank lines. Frontmatter is handled separately by _split_frontmatter."""
    text = re.split(r"\n#{1,6}\s*related articles\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)   # markdown links -> anchor text
    text = re.sub(r"https?://\S+", "", text)               # bare URLs
    text = re.sub(r"^_last updated:.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_sections(body: str) -> list[tuple[str, str]]:
    """Split a cleaned body into (section_path, text) chunks.

    Header-aware: sections come from markdown headers (with their hierarchy as a
    breadcrumb path); small sections are merged up to TARGET_WORDS, and sections
    over MAX_WORDS are window-split with OVERLAP_WORDS so nothing is truncated."""
    matches = list(_HEADER_RE.finditer(body))
    raw_sections: list[tuple[str, str]] = []
    if not matches:
        if body.strip():
            raw_sections.append(("", body.strip()))
    else:
        pre = body[:matches[0].start()].strip()
        if pre:
            raw_sections.append(("", pre))
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level, heading = len(m.group(1)), m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            body_text = body[start:end].strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            # Keep the heading inside the text so its words survive section merges.
            section_text = f"{heading}\n{body_text}".strip() if body_text else heading
            raw_sections.append((" > ".join(h for _, h in stack), section_text))

    chunks: list[tuple[str, str]] = []
    cur_path, cur = "", []
    for path, text in raw_sections:
        words = text.split()
        if len(words) > MAX_WORDS:
            if cur:
                chunks.append((cur_path, " ".join(cur)))
                cur_path, cur = "", []
            step = max(1, TARGET_WORDS - OVERLAP_WORDS)
            for i in range(0, len(words), step):
                chunks.append((path, " ".join(words[i:i + TARGET_WORDS])))
                if i + TARGET_WORDS >= len(words):
                    break
            continue
        if cur and len(cur) + len(words) > TARGET_WORDS:
            chunks.append((cur_path, " ".join(cur)))
            cur_path, cur = "", []
        if not cur:
            cur_path = path
        cur += words
    if cur:
        chunks.append((cur_path, " ".join(cur)))
    return chunks


def _doc_chunks(path: Path) -> list[tuple[str, list[str]]]:
    """Return (content, tokens) per chunk for one markdown file. `content` is
    shown to the LLM; `tokens` (enriched with title/breadcrumbs/filename) feed BM25."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    frontmatter, body = _split_frontmatter(raw)
    title = _extract_title(frontmatter, body)
    breadcrumbs = _extract_breadcrumbs(frontmatter)
    body = clean_content(body)
    slug = re.sub(r"^\d+[-_]?", "", path.stem).replace("-", " ").replace("_", " ")
    enrichment = " ".join([title, " ".join(breadcrumbs), slug])

    out = []
    for section_path, text in _chunk_sections(body):
        prefix = section_path or title
        content = f"{prefix}\n{text}".strip() if prefix else text
        tokens = _tokenize(f"{content} {enrichment}")
        if tokens:
            out.append((content, tokens))
    return out


def build_indexes():
    """Build a chunk-level BM25 index per product area."""
    INDEXES.clear()
    for product, path in CORPUS_PATHS.items():
        if path is None or not path.exists():
            continue

        files, contents, tokenized = [], [], []
        for f in path.rglob("*.md"):
            if f.name in EXCLUDED_FILES and not (f.name == "support.md" and f.parent != path):
                continue
            try:
                for content, tokens in _doc_chunks(f):
                    files.append(f)
                    contents.append(content)
                    tokenized.append(tokens)
            except Exception:
                continue

        if tokenized:
            INDEXES[product] = (BM25Okapi(tokenized), files, contents)


def retrieve(query: str, product_area: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """Retrieve up to top_k relevant chunks for the query from the product area.

    Applies a relative relevance floor (drops the long tail of marginal hits) and
    caps chunks per document so one file can't fill the result set."""
    if not INDEXES:
        build_indexes()

    area = product_area.strip().lower()
    if area == "none" or area not in INDEXES:
        return []

    bm25, files, contents = INDEXES[area]
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(scores, range(len(files))), key=lambda t: (t[0], t[1]), reverse=True)
    if not ranked or ranked[0][0] <= 0.0:
        return []

    floor = RELATIVE_FLOOR * ranked[0][0]
    results, per_doc = [], {}
    for score, idx in ranked:
        if score <= 0.0 or score < floor:
            break
        f = files[idx]
        if per_doc.get(f, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        try:
            rel_path = str(f.relative_to(ROOT)).replace("\\", "/")
        except Exception:
            continue
        per_doc[f] = per_doc.get(f, 0) + 1
        results.append({"path": rel_path, "content": contents[idx], "score": float(score)})
        if len(results) >= top_k:
            break
    return results
