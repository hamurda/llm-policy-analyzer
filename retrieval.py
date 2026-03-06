import json, re, torch, logging
from sentence_transformers import SentenceTransformer, util

from config import (
    EMBEDDING_MODEL, SIM_THRESHOLD, ARTICLES,
    GDPR_PARAGRAPHS_JSONL, GDPR_RULES_EMBEDDING_JSON,
    GDPR_CHECKLIST_EMBEDDING_JSON,
)

logger = logging.getLogger(__name__)

# helpers
def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# module-level singletons (loaded once)
_encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
_rules   = [r for r in _load_json(GDPR_RULES_EMBEDDING_JSON)
            if str(r.get("article")) in {str(a) for a in ARTICLES}]
_checks  = _load_json(GDPR_CHECKLIST_EMBEDDING_JSON)
_paras   = [p for p in _load_jsonl(GDPR_PARAGRAPHS_JSONL)
            if p.get("structure", {}).get("article") in ARTICLES]

# policy chunking
def _chunk_policy(policy_text: str, min_chars: int = 80) -> list[str]:
    """
    Split raw policy text into meaningful chunks.
    """
    # try double-newline first (most policies use blank lines between sections)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", policy_text) if c.strip()]

    # fallback: if that produces fewer than 3 chunks, split on single newlines
    if len(chunks) < 3:
        chunks = [c.strip() for c in policy_text.split("\n") if c.strip()]

    # drop fragments too short to carry meaning
    chunks = [c for c in chunks if len(c) >= min_chars]
    return chunks

# chunk-article matching
def _match_chunks_to_articles(chunks: list[str], threshold: float = SIM_THRESHOLD) -> dict[str, list[str]]:
    """
    For each policy chunk, find which GDPR articles it relates to.

    Returns { article_number_str: [chunk_text, ...] }
    A chunk can match multiple articles; an article can collect many chunks.
    """
    chunk_embs = _encoder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)

    # pre-stack rule embeddings once
    rule_tensors = []
    for rule in _rules:
        embs = rule.get("embeddings")
        if not embs:
            rule_tensors.append(None)
            continue
        rule_tensors.append(
            torch.stack([torch.tensor(e, dtype=torch.float32) for e in embs])
        )

    article_chunks: dict[str, list[str]] = {}

    for chunk_idx, chunk_emb in enumerate(chunk_embs):
        for rule, rule_t in zip(_rules, rule_tensors):
            if rule_t is None:
                continue
            score = float(util.cos_sim(chunk_emb, rule_t).max())
            if score >= threshold:
                art = str(rule["article"])
                article_chunks.setdefault(art, []).append(chunks[chunk_idx])

    return article_chunks

# public API
def match_policy_to_articles(policy_text: str, threshold: float = SIM_THRESHOLD) -> list[dict]:
    """
    Given raw policy text, chunk it, match chunks to GDPR articles,
    and return a list of article dicts ready for LLM evaluation.

    Each dict:
      { article, title, intent, severity,
        subpoints: [{clause, text, checklist}],
        policy_text: <only the matched chunks, joined>,
        gdpr_paragraphs: [{official_cite, text}] }
    """
    chunks = _chunk_policy(policy_text)
    logger.info(f"Policy split into {len(chunks)} chunks")

    article_chunks = _match_chunks_to_articles(chunks, threshold)
    logger.info(
        f"Chunks matched to {len(article_chunks)} articles: "
        f"{sorted(article_chunks.keys(), key=lambda x: int(x))}"
    )

    # build result for every in-scope article
    art_set = {str(a) for a in ARTICLES}
    results = []

    for chk in _checks:
        art = str(chk["article"])
        if art not in art_set:
            continue

        # subpoints & checklist
        subpoints = [
            {"clause": r["clause"], "text": r["text"],
             "checklist": r.get("checklist", [])}
            for r in chk.get("requirements", [])
        ]

        # GDPR law paragraphs
        law_paras = [
            {"official_cite": p["official_cite"], "text": p["text"]}
            for p in _paras
            if p["structure"]["article"] == int(art)
        ]

        # matched policy excerpts (or empty if no chunks matched)
        matched = article_chunks.get(art, [])

        results.append({
            "article":         art,
            "title":           chk.get("title", ""),
            "intent":          chk.get("intent", ""),
            "severity":        chk.get("severity", ""),
            "subpoints":       subpoints,
            "policy_text":     "\n\n".join(matched) if matched else "",
            "gdpr_paragraphs": law_paras,
        })

    results.sort(key=lambda x: int(x["article"]))
    return results