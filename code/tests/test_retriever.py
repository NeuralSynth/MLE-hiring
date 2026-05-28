import pytest
from pathlib import Path
from retriever import (
    retrieve, EXCLUDED_FILES, tokenize, clean_content, _chunk_sections,
    embed_texts, _fuse_score, BM25_WEIGHT, EMBED_WEIGHT,
)

def test_retrieves_devplatform_results():
    results = retrieve("how do I reset my test", "devplatform")
    assert len(results) > 0

def test_retrieves_visa_results():
    results = retrieve("exchange rate for travel", "visa")
    assert len(results) > 0

def test_retrieves_claude_results():
    results = retrieve("API rate limits", "claude")
    assert len(results) > 0

def test_none_searches_all_areas():
    # L3: 'none' no longer blinds retrieval — an on-topic query still finds docs.
    results = retrieve("how do I reset my password", "none")
    assert len(results) > 0

def test_no_lexical_match_returns_empty():
    assert retrieve("zzzznomatchxyzzy plughquux", "visa") == []

def test_all_paths_exist_on_disk():
    """Critical — source_documents paths must be real files."""
    results = retrieve("account access problem", "devplatform")
    for r in results:
        assert Path(r["path"]).exists(), f"Path does not exist: {r['path']}"

def test_results_have_required_keys():
    results = retrieve("billing question", "visa")
    for r in results:
        assert "path" in r
        assert "content" in r
        assert "score" in r

def test_scores_are_descending():
    results = retrieve("delete account", "devplatform")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_different_queries_return_different_results():
    r1 = retrieve("delete account", "devplatform")
    r2 = retrieve("API rate limit exceeded", "devplatform")
    assert r1[0]["path"] != r2[0]["path"]

def test_no_excluded_files_in_results():
    """Trap files must never appear in results."""
    for area in ["devplatform", "claude", "visa"]:
        results = retrieve("account billing support help", area)
        for r in results:
            fname = Path(r["path"]).name
            assert fname not in EXCLUDED_FILES, f"Excluded file returned: {r['path']}"

def test_no_yaml_frontmatter_in_content():
    """YAML frontmatter must be stripped from returned content."""
    for area in ["devplatform", "claude", "visa"]:
        results = retrieve("support documentation", area)
        for r in results:
            assert not r["content"].strip().startswith("---"), f"Frontmatter in: {r['path']}"

def test_visa_query_returns_specific_doc():
    """Visa tickets must not all cite generic support.md."""
    results = retrieve("Visa card transaction declined at merchant terminal", "visa")
    if results:
        top_path = Path(results[0]["path"]).name
        assert top_path != "support.md"


# --- Tokenizer (punctuation-proof, stopword-free) ---

def test_tokenizer_splits_punctuation_and_drops_stopwords():
    assert tokenize("What are the rate-limits? Reset my API.") == ["rate", "limits", "reset", "api"]

def test_tokenizer_keeps_alphanumeric_identifiers():
    assert tokenize("error 429 on endpoint v2") == ["error", "429", "endpoint", "v2"]


# --- Content cleaning ---

def test_clean_content_strips_related_articles_and_urls():
    raw = "Real answer text here.\n\n## Related Articles\n- [Other thing](https://x.com/a)\n- [More](https://y.com)"
    out = clean_content(raw)
    assert "Related Articles" not in out
    assert "Other thing" not in out
    assert "http" not in out
    assert "Real answer text here." in out

def test_clean_content_keeps_link_anchor_drops_url():
    out = clean_content("See the [API docs](https://platform.example.com/docs) for details")
    assert "API docs" in out
    assert "http" not in out


# --- Header-aware chunking ---

def test_short_section_is_one_chunk():
    chunks = _chunk_sections("# Title\nA short answer of only a few words.")
    assert len(chunks) == 1

def test_long_section_is_split_with_size_cap():
    body = "# Guide\n" + " ".join(f"word{i}" for i in range(900))
    chunks = _chunk_sections(body)
    assert len(chunks) > 1
    assert all(len(text.split()) <= 320 for _, text in chunks)

def test_header_words_preserved_through_merge():
    body = "# Guide\nintro\n## Billing\nbilling text\n### Refunds\nrefund details"
    blob = " ".join(text for _, text in _chunk_sections(body))
    assert "Billing" in blob and "Refunds" in blob

def test_deep_section_keeps_ancestor_path():
    big = " ".join(f"w{i}" for i in range(260))  # > TARGET so it isn't merged away
    body = f"# Guide\nintro\n## Billing\nshort\n### Refunds\n{big}"
    paths = [p for p, _ in _chunk_sections(body)]
    assert any("Billing" in p and "Refunds" in p for p in paths)


# --- Relevance gate ---

def test_results_within_relative_floor():
    results = retrieve("exchange rate for travel abroad", "visa")
    if len(results) > 1:
        top = results[0]["score"]
        assert all(r["score"] >= 0.25 * top for r in results)


# --- Semantic re-rank (optional; gracefully degrades to BM25) ---

def test_fuse_score_weights():
    assert _fuse_score(1.0, 0.0) == pytest.approx(BM25_WEIGHT)
    assert _fuse_score(0.0, 1.0) == pytest.approx(EMBED_WEIGHT)
    assert _fuse_score(1.0, 1.0) == pytest.approx(1.0)

def test_fuse_score_clamps_negative_cosine():
    # a negative (dissimilar) cosine never subtracts from the BM25 signal
    assert _fuse_score(0.5, -0.9) == pytest.approx(BM25_WEIGHT * 0.5)

def test_embed_texts_none_when_disabled():
    # conftest sets DISABLE_EMBEDDINGS, so the suite runs BM25-only.
    assert embed_texts(["anything"]) is None
