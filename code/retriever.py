import os
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

# Ranking / gate.
RELATIVE_FLOOR = 0.25      # drop hits scoring below this fraction of the top hit
MAX_CHUNKS_PER_DOC = 2     # don't let one file fill the result set
CANDIDATES = 20            # BM25 candidate pool size handed to the semantic re-rank
BM25_WEIGHT = 0.6          # fusion weights (lexical-leaning: BM25 nails exact terms)
EMBED_WEIGHT = 0.4

INDEXES = {}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords. Used for both the
    index and the query so matching stays symmetric (and punctuation-proof)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def format_context(chunks: list) -> str:
    """Render retrieved chunks as a numbered, path-labelled block for an LLM
    prompt. Shared by the escalation supervisor and the response generator."""
    return "\n\n".join(
        f"Document {i + 1} (Path: {doc['path']}):\n{doc['content']}"
        for i, doc in enumerate(chunks)
    )


# ---------------------------------------------------------------------------
# Optional semantic embeddings (model2vec). Everything degrades to pure BM25 if
# the library/model isn't available or DISABLE_EMBEDDINGS is set.
# ---------------------------------------------------------------------------

_EMBEDDER = None
_EMBEDDER_LOADED = False


def _get_embedder():
    global _EMBEDDER, _EMBEDDER_LOADED
    if _EMBEDDER_LOADED:
        return _EMBEDDER
    _EMBEDDER_LOADED = True
    if os.getenv("DISABLE_EMBEDDINGS"):
        return None
    try:
        from model2vec import StaticModel
        _EMBEDDER = StaticModel.from_pretrained(os.getenv("EMBED_MODEL", "minishlab/potion-base-8M"))
    except Exception:
        _EMBEDDER = None
    return _EMBEDDER


def embed_texts(texts):
    """Return an (n, d) matrix of L2-normalized embeddings, or None if unavailable."""
    model = _get_embedder()
    if model is None:
        return None
    try:
        import numpy as np
        vecs = np.asarray(model.encode(list(texts)), dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
    except Exception:
        return None


def _fuse_score(bm25_norm: float, cosine: float) -> float:
    """Combine a normalized BM25 score with a (clamped) cosine similarity."""
    return BM25_WEIGHT * bm25_norm + EMBED_WEIGHT * max(0.0, cosine)


# ---------------------------------------------------------------------------
# Content cleaning + chunking
# ---------------------------------------------------------------------------

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
        tokens = tokenize(f"{content} {enrichment}")
        if tokens:
            out.append((content, tokens))
    return out


def build_indexes():
    """Build a chunk-level BM25 index (and optional embedding matrix) per area."""
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
            embeddings = embed_texts(contents)  # None if the embedder is unavailable
            INDEXES[product] = (BM25Okapi(tokenized), files, contents, embeddings)


def _score_area(area: str, query_tokens: list, query_text: str) -> list:
    """Rank one area's chunks: BM25 candidate pool, optional semantic re-rank,
    relative floor. Returns [(fused_score, area, chunk_idx), ...]. Scores are
    normalized within the area so cross-area results are roughly comparable."""
    bm25, files, contents, embeddings = INDEXES[area]
    scores = bm25.get_scores(query_tokens)
    order = sorted(range(len(files)), key=lambda i: (scores[i], i), reverse=True)
    candidates = [i for i in order if scores[i] > 0.0][:CANDIDATES]
    if not candidates:
        return []
    top_bm25 = scores[candidates[0]]

    cos = None
    if embeddings is not None:
        qv = embed_texts([query_text])
        if qv is not None:
            cos = embeddings[candidates] @ qv[0]  # cosine (vectors are L2-normalized)

    fused = []
    for rank, i in enumerate(candidates):
        bm_norm = scores[i] / top_bm25 if top_bm25 > 0 else 0.0
        if cos is not None:
            fused.append((_fuse_score(bm_norm, float(cos[rank])), area, i))
        else:
            fused.append((bm_norm, area, i))
    fused.sort(key=lambda t: (-t[0], t[2]))

    floor = RELATIVE_FLOOR * fused[0][0]
    return [t for t in fused if t[0] >= floor and t[0] > 0.0]


def _search_all(query_tokens: list, query_text: str, exclude: str = None) -> list:
    out = []
    for a in INDEXES:
        if a == exclude:
            continue
        out.extend(_score_area(a, query_tokens, query_text))
    return out


def retrieve(query: str, product_area: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """Retrieve up to top_k relevant chunks.

    L3: search the classified area first, but fall back to all areas when the
    area is unknown / "none" or yields nothing, so a misclassification (or a
    "none" classification) doesn't blind retrieval. Within an area, BM25 supplies
    candidates that an optional semantic model re-ranks; a relative floor drops
    the marginal tail and chunks-per-document are capped."""
    if not INDEXES:
        build_indexes()

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    area = product_area.strip().lower()
    if area in INDEXES and area != "none":
        scored = _score_area(area, query_tokens, query)
        if not scored:
            scored = _search_all(query_tokens, query, exclude=area)
    else:
        scored = _search_all(query_tokens, query)

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    results, per_doc = [], {}
    for score, a, i in scored:
        _, files, contents, _ = INDEXES[a]
        f = files[i]
        if per_doc.get(f, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        try:
            rel_path = str(f.relative_to(ROOT)).replace("\\", "/")
        except Exception:
            continue
        per_doc[f] = per_doc.get(f, 0) + 1
        results.append({"path": rel_path, "content": contents[i], "score": round(float(score), 4)})
        if len(results) >= top_k:
            break
    return results
