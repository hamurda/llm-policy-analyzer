from src.retrieval.rule_matcher import match_rules, match_subitems, retrieve_paragraphs_for_articles
from src.retrieval.retrieval_utils import gdpr_rules, all_checklists, policy_chunks


def build_policy_excerpt(policy: dict) -> str:
    return f"{policy.get('structure', {}).get('heading', '')} {policy.get('text', '')}"

def match_articles(policy_excerpt: str, gdpr_rules: list) -> set[int]:
    """Return a set of article numbers that match this policy excerpt."""
    articles = match_rules(policy_excerpt, gdpr_rules)
    return {a["article"] for a in articles}

def filter_and_enrich_checklists(article_numbers: set[int], all_checklists: list, policy_excerpt: str) -> list[dict]:
    """Filter checklists to the matched articles and enrich with subpoints."""
    filtered = [c for c in all_checklists if c["article"] in article_numbers]
    for rule in filtered:
        rule["subpoints"] = match_subitems(policy_excerpt, rule, 0.2)
    return filtered

def process_policy(policy: dict, gdpr_rules: list, all_checklists: list) -> dict:
    """Process one policy chunk into its matched rules + GDPR paragraphs."""
    policy_excerpt = build_policy_excerpt(policy)
    article_numbers = match_articles(policy_excerpt, gdpr_rules)
    filtered_checklist = filter_and_enrich_checklists(article_numbers, all_checklists, policy_excerpt)
    paragraphs = retrieve_paragraphs_for_articles(policy_excerpt, filtered_checklist, top_k_per_article=10)

    return {
        "policy_chunk": {
            "id": policy["id"],
            "cite": policy["cite"],
            "structure": policy["structure"],
            "text": policy["text"]
        },
        "article_numbers": list(article_numbers),
        "matched_articles": filtered_checklist,
        "paragraphs": paragraphs,
    }

def build_rule_map(policy_chunks: list=policy_chunks, gdpr_rules: list=gdpr_rules, all_checklists: list=all_checklists) -> list[dict]:
    return [process_policy(p, gdpr_rules, all_checklists) for p in policy_chunks]

def chunks_to_articles(processed_policy_chunks:list):
    article_view = {}

    for entry in processed_policy_chunks:
        chunk = entry["policy_chunk"]
        for ma in entry.get("matched_articles", []):
            art = ma["article"]

            if art not in article_view:
                article_view[art] = {
                    "article": art,
                    "title": ma.get("title"),
                    "intent": ma.get("intent"),
                    "severity": ma.get("severity"),
                    "subpoints": ma.get("subpoints", []),
                    "policy_evidence": []
                }

            article_view[art]["policy_evidence"].append({
                "id": chunk["id"],
                "cite": chunk["cite"],
                "heading": chunk["structure"]["heading"],
                "text": chunk["text"]
            })

    return list(article_view.values())