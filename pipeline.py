import logging, re
from retrieval import match_policy_to_articles
from llm_client import evaluate_article

logger = logging.getLogger(__name__)


def analyze_policy(policy_text: str) -> list[dict]:
    """
    End-to-end: raw policy text in → list of ArticleResult dicts out.
    One LLM call per matched article.
    """
    if not policy_text or not policy_text.strip():
        return []

    logger.info("Matching policy to GDPR articles …")
    articles = match_policy_to_articles(policy_text)
    logger.info(f"Matched {len(articles)} articles")

    results = []
    for art in articles:
        logger.info(f"Evaluating Article {art['article']} – {art['title']} …")
        result = evaluate_article(art)
        results.append(result)

    # sort by article number (LLM sometimes returns "Article 5" instead of "5")
    def _art_num(x):
        m = re.search(r"\d+", str(x.get("article", "0")))
        return int(m.group()) if m else 0

    results.sort(key=_art_num)
    return results
