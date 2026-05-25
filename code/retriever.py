from pathlib import Path
from rank_bm25 import BM25Okapi
from config import CORPUS_PATHS, BM25_TOP_K, ROOT

INDEXES = {}

def build_indexes():
    """Build BM25 indexes for devplatform, claude, and visa directories."""
    for product, path in CORPUS_PATHS.items():
        if path is None:
            continue
        if not path.exists():
            continue
        
        files = list(path.rglob("*.md"))
        tokenized_corpus = []
        valid_files = []
        
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                # Simple tokenization: lowercase and split by whitespace
                tokens = content.lower().split()
                tokenized_corpus.append(tokens)
                valid_files.append(f)
            except Exception:
                continue
                
        if tokenized_corpus:
            INDEXES[product] = (BM25Okapi(tokenized_corpus), valid_files)

def retrieve(query: str, company: str) -> list[dict]:
    """Retrieve top_k matching documents for the query from the specific company's index."""
    company_lower = company.strip().lower()
    if company_lower == "none" or company_lower not in INDEXES:
        return []
        
    index, files = INDEXES[company_lower]
    query_tokens = query.lower().split()
    scores = index.get_scores(query_tokens)
    
    # Pair scores with files, sort descending, and take top BM25_TOP_K
    ranked = sorted(zip(scores, files), reverse=True)[:BM25_TOP_K]
    
    results = []
    for s, f in ranked:
        if s > 0.0:  # Only return documents with a positive match score
            try:
                rel_path = str(f.relative_to(ROOT)).replace("\\", "/")
                content = f.read_text(encoding="utf-8", errors="ignore")
                results.append({
                    "path": rel_path,
                    "content": content,
                    "score": float(s)
                })
            except Exception:
                continue
    return results
