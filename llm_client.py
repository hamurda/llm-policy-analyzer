import json, os, time, logging, openai
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from config import OPENAI_MODEL

logger = logging.getLogger(__name__)
load_dotenv()

# Pydantic schema
class ArticleResult(BaseModel):
    article: str    = Field(description="Article number")
    title: str      = Field(description="Article heading")
    coverage: Literal["Covered", "Partial", "Not Observed"] = Field(
        description="Overall coverage verdict based on the policy evidence.")
    applicable: bool = Field(
        description="Does this article apply in practice to this specific company/policy?")
    rationale: str   = Field(
        description="2-3 sentence explanation.  If Partial, state what is missing.")
    policy_citations: list[str] = Field(
        description="Minimal verbatim phrases from the policy that justify the verdict.")
    gdpr_citations: list[str] = Field(
        description="Official GDPR paragraph references, e.g. 'Art 13(1)(c)'.")

# prompt builder
def _build_prompt(article: dict) -> str:
    # Format subpoints + checklist into a readable block
    subpoint_block = ""
    for sp in article.get("subpoints", []):
        questions = "\n".join(f"      - {q}" for q in sp.get("checklist", []))
        subpoint_block += (
            f"    {sp['clause']}: {sp['text']}\n"
            f"      Checklist:\n{questions}\n"
        )

    # Format GDPR law paragraphs
    law_block = "\n".join(
        f"  {p['official_cite']}: {p['text']}"
        for p in article.get("gdpr_paragraphs", [])
    )

    return f"""You are a GDPR compliance analyst.  Evaluate the privacy policy below
against GDPR Article {article['article']} – {article['title']}.

Use ONLY the provided policy text and GDPR paragraphs.  Do NOT invent facts.

═══ ARTICLE {article['article']} – {article['title']} ═══
Intent: {article.get('intent', '')}
Severity: {article.get('severity', '')}

Subpoints & checklist questions:
{subpoint_block}

═══ RELEVANT GDPR TEXT ═══
{law_block}

═══  POLICY EXCERPTS (retrieved sections relevant to this article) ═══
{article['policy_text'][:12000]}

═══ TASK ═══
1. Use the checklist questions above as a guide.
2. Search the policy text for evidence relevant to each subpoint.
3. Decide overall coverage:
   • "Covered"      – policy clearly addresses the article's requirements
   • "Partial"      – some requirements addressed, some missing
   • "Not Observed" – no meaningful evidence found
4. Set applicable=true if the article applies in principle (even if not evidenced).
5. Cite minimal policy phrases and official GDPR references.
6. Write a 2-3 sentence rationale. If Partial, state what is missing.

Return ONLY valid JSON matching the provided schema.
"""

#LLM call with retries
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def evaluate_article(article: dict, max_retries: int = 3) -> dict:
    """Call OpenAI for one article.  Returns dict matching ArticleResult."""
    prompt = _build_prompt(article)
    client = _get_client()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.parse(
                model=OPENAI_MODEL,
                instructions="You are a GDPR compliance reasoning assistant.",
                input=[{"role": "user", "content": prompt}],
                text_format=ArticleResult,
            )

            result = json.loads(response.output_parsed.model_dump_json())
            logger.info(f"Art {article['article']}: {result.get('coverage')}")
            return result

        except json.JSONDecodeError as e:
            logger.warning(
                f"Art {article['article']} attempt {attempt}: bad JSON – {e}"
            )
        except Exception as e:
            logger.warning(
                f"Art {article['article']} attempt {attempt}: {type(e).__name__} – {e}"
            )

        if attempt < max_retries:
            time.sleep(2 ** attempt)          # exponential back-off

    # all retries failed → return a safe fallback
    logger.error(f"Art {article['article']}: all {max_retries} attempts failed")
    return {
        "article":          article["article"],
        "title":            article["title"],
        "coverage":         "Not Observed",
        "applicable":       True,
        "rationale":        "Evaluation failed after multiple retries – manual review needed.",
        "policy_citations": [],
        "gdpr_citations":   [],
    }
