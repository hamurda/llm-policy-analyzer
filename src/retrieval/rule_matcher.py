import torch
from typing import List, Dict, Any
from sentence_transformers import util

from src.retrieval.retrieval_utils import st_embedding, gdpr_db, policy_db
from src.utils.config import SIM_THRESHOLD, TOP_K_PER_ARTICLE


def match_rules(policy_excerpt: str, rules: List[Dict[str, Any]], top_n: int = 10, threshold: float = SIM_THRESHOLD):
    """
    Return sorted list of matched rules with scores.
    {
        "article": str,
        "title": str,
        "severity": str,
        "intent": str,
        "score": float,
    }
    """
    print("Retrieving target rules...")
    q_emb = st_embedding.encode(policy_excerpt, convert_to_tensor=True)
    matches = []
    for r in rules:
        rule_emb = r.get("embeddings")
        if rule_emb is None:
            continue
        rule_emb = torch.stack([torch.tensor(e, dtype=torch.float32) for e in r["embeddings"]])
        score = util.cos_sim(q_emb, rule_emb).item()
        if score >= threshold:
            matches.append({
                    "article": r["article"],
                    "title": r["title"],
                    "severity": r["severity"],
                    "intent": r["intent"],
                    "score": float(score),
                })
    
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)

    unique_matches = []
    seen_articles = set()

    for match in matches:
        if match["article"] not in seen_articles:
            unique_matches.append(match)
            seen_articles.add(match["article"])
            if len(unique_matches)>=top_n:
                break
                  
    return unique_matches

def match_subitems(policy_excerpt: str, rule: Dict[str, Any], threshold: float = SIM_THRESHOLD):
    """
    Return sorted list of matched subitems with scores.
    {
        "clause": str,
        "text": str,
        "checklist": str,
        "score": float,
    }
    """
    print("Retrieving target subpoints...")
    q_emb = st_embedding.encode(policy_excerpt, convert_to_tensor=True)
    matches = []
    for clause in rule.get("requirements", []):
        clause_emb = clause["embeddings"]
        if clause_emb is None:
            continue
        clause_emb = torch.stack([torch.tensor(e, dtype=torch.float32) for e in clause["embeddings"]])
        score = util.cos_sim(q_emb, clause_emb).item()
        if score >= threshold:
            matches.append({
                    "clause": clause["clause"],
                    "text": clause["text"],
                    "checklist": clause["checklist"],
                    "score": float(score),
                })
    
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)

    unique_matches = []
    seen_clauses = set()

    for match in matches:
        if match["clause"] not in seen_clauses:
            unique_matches.append(match)
            seen_clauses.add(match["clause"])
                  
    return unique_matches

def retrieve_paragraphs_for_articles(policy_excerpt: str, candidate_articles: List[str], top_k_per_article: int = TOP_K_PER_ARTICLE):
    """
    For each candidate article, restrict search to paragraphs whose metadata article == candidate.
    Returns dict article->list[paragraphs]
    {
        "article": str,
        "title": str,
        "text": str,
        "cite": str,
        "chunk_id": str,
    }
    """
    print("Retrieveing relevant paragraphs...")
    results_by_article = []

    for art in candidate_articles:
        art_paras = gdpr_db.similarity_search(
            policy_excerpt,
            k=top_k_per_article,
            filter={"article": int(art["article"])},
        )
        if not art_paras:
            continue

        for para in art_paras:
            p = {
                "article": para.metadata.get("article"),
                "title": para.metadata.get("title"),
                "text":para.page_content,
                "cite": para.metadata.get("chunk_cite", ""),
                "chunk_id":para.metadata.get("chunk_id", "")
            }
            results_by_article.append(p)
            
    return results_by_article

def retrieve_paragraphs_for_subitems(policy_excerpt: str, subitems: List[str], top_k_per_article: int = TOP_K_PER_ARTICLE):
    """
    For each candidate article, restrict search to paragraphs whose metadata article == candidate.
    Returns dict article->list[paragraphs]
    {
        "article": str,
        "title": str,
        "text": str,
        "cite": str,
        "chunk_id": str,
    }
    """
    print("Retrieveing relevant subitems...")
    results_by_article = []

    for item in subitems:
        art_paras = gdpr_db.similarity_search(
            policy_excerpt,
            k=top_k_per_article,
            filter={"official_cite": f"Reg (EU) 2016/679, Art {item["clause"]}"}, # "official_cite": "Reg (EU) 2016/679, Art 6(1)(d)"
        )
        if not art_paras:
            continue

        for para in art_paras:
            p = {
                "article": para.metadata.get("article"),
                "title": para.metadata.get("title"),
                "text":para.page_content,
                "cite": para.metadata.get("chunk_cite", ""),
                "chunk_id":para.metadata.get("chunk_id", "")
            }
            results_by_article.append(p)
            
    return results_by_article
