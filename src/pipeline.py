from src.retrieval.policy_retriever import build_rule_map, chunks_to_articles
from src.llm_client import build_article_prompt, call_llm_for_structured_result_gemini, build_subpoint_prompt, SubpointResult
from src.retrieval.retrieval_utils import get_gdpr_paragraphs_by_article
from src.utils.helpers import write_to_json

def evaluate_articles():   
    rule_map = build_rule_map()
    article_map = chunks_to_articles(rule_map)

    compiled_result = []

    for article in article_map:
        law_paras = get_gdpr_paragraphs_by_article(int(article.get("article", "")))

        sub_results = []
        for sub in article["subpoints"]:
            prompt = build_subpoint_prompt(article, sub, article["policy_evidence"])
            result = call_llm_for_structured_result_gemini(prompt, SubpointResult)
            sub_results.append(result)

        prompt = build_article_prompt(article, law_paras)
        result = call_llm_for_structured_result_gemini(prompt)
        result["subpoints"] = sub_results

        compiled_result.append(result)

    write_to_json("verdict.json", compiled_result) #will be ui later.
