import os
import json
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Any, List
from dotenv import load_dotenv

from src.utils.config import GEMINI_MODEL


load_dotenv(override=True)
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

gemini_api_key = os.environ['GEMINI_API_KEY']  #2 models to compare output, remove one later
openai_api_key = os.environ['OPENAI_API_KEY']

class Result(BaseModel):
    article:str = Field(description="Article number")
    title:str = Field(description="Article heading")
    coverage: Literal["Covered", "Partial", "Not Observed"] = Field(description="Decided coverage strictly based on the snippets.")
    applicable: bool = Field(description="Given this specific company/policy, does this article actually matter? Does this artical actually apply in practice? Can be True or False")
    relevant: bool = Field(description="Does this GDPR article conceptually apply to the domain/industry/business type? Could the article ever apply? If yes, relevant is True.")
    rationale: str = Field(description="Short explanation of the decision for the coverage.")

class SubpointResult(BaseModel):
    clause:str = Field(description="Clause number")
    coverage: Literal["Covered", "Partial", "Not Observed"] = Field(description="Decided coverage strictly based on the snippets.")
    checklist_questions: List[str] = Field(description="Questions in the checklist. Put the question here and the answer to 'question_answer'")
    checklist_answers: List[str] = Field(description="Find the exact answer for each question in the provided checklist and put the brief answer against it.")
    policy_citations: list[str] = Field(description="Exact snippet texts that decision for the coverage is based on.")
    rationale: str = Field(description="Short explanation of the decision for the coverage.")


def build_subpoint_prompt(article_obj, subpoint, policy_excerpts):
    return f"""
        You are evaluating a privacy policy against GDPR requirements.

        Article: {article_obj["article"]} - {article_obj["title"]}
        Subpoint: {subpoint["clause"]} - {subpoint["text"]}
        Intent: {article_obj["intent"]}

        Checklist:
        {chr(10).join(f"- {q}" for q in subpoint["checklist"])}

        Policy Excerpts:
        {chr(10).join(f"- {c}" for c in policy_excerpts)}

        Instructions:
        1. Answer each checklist question with "Yes", "No", or "Partial", and justify your answer by quoting relevant policy text (if available).
        2. Determine overall coverage of this subpoint as one of: "Covered", "Partial", or "Not Observed".
        3. Return JSON only in the provided schema.

        """

def build_article_prompt(article_obj, law_paragraphs):
    """
    article_obj: dict in your schema (article, title, intent, severity, subpoints, policy_evidence)
    law_paragraphs: list of dicts for this article, each with 'official_cite' and 'text'
    """
    # Format law text
    law_blocks = [
        f"{para['official_cite']}: {para['text']}"
        for para in law_paragraphs
    ]

    # Format policy evidence
    policy_blocks = [
        f"{ev['cite']}:\n{ev['text']}"
        for ev in article_obj.get("policy_evidence", [])
    ]

    # Rules summary (intent, severity, subpoints)
    flagged_rules_brief = {
        "intent": article_obj.get("intent", ""),
        "severity": article_obj.get("severity", ""),
        "subpoints": [
            {
                "clause": sp["clause"],
                "text": sp["text"],
                "checklist": sp.get("checklist", [])
            }
            for sp in article_obj.get("subpoints", [])
        ]
    }

    prompt = f"""
    You are a GDPR compliance assistant. ONLY use the provided context (policy evidence plus the law text below).
    Do NOT invent or assume facts not present in the supplied text.

    ARTICLE {article_obj['article']} – {article_obj['title']}

    Flagged Rules (business logic for this article):
    {json.dumps(flagged_rules_brief, indent=2)}

    Relevant GDPR Text (only the paragraphs below are permitted for citation):
    {chr(10).join(law_blocks)}

    Policy Evidence (retrieved excerpts relevant to this article):
    {chr(10).join(policy_blocks)}

    TASK:
    For each subpoint in the Article's `subpoints`, perform an assessment:
    - article: the article number (string)
    - title: the official title of the article
    - coverage: "Covered" | "Partial" | "Not Observed"
    - applicable: true/false   # does this article apply in principle? if it applies but is not evidenced, set true and coverage "Not Observed"
    - relevant: true/false     # is this article conceptually relevant to the business/domain?
    - policy_citations: list of minimal phrases from the Policy Evidence that justify your decision (short snippets, not whole paragraphs)
    - gdpr_citations: list of exact paragraph/subpoint citations from the provided GDPR text, like "Reg (EU) 2016/679, Art 13(1)(c)"
    - rationale: short explanation (max 3 sentences. If it's Partial, always indicate what is missing.)

    Use the checklist items provided in each subpoint as guiding questions for your assessment.

    After evaluating all subpoints:
    - Roll up the results into a single Article-level object with:
      - article
      - title
      - coverage: aggregate (if all covered → "Covered"; if mix → "Partial"; if none → "Not Observed")
      - applicable: true if at least one subpoint is applicable
      - relevant: true if the Article as a whole conceptually applies to the policy domain
      - policy_citations: union of all unique minimal phrases from subpoints
      - gdpr_citations: union of all unique citations from subpoints
      - rationale: max 3 sentences summarizing the overall assessment

    Important rules for output:
    1) Use only the GDPR text above. Cite the exact paragraph/subpoint (use 'cite' present in the law text metadata).
    2) policy_citations should be minimal — short phrases or sentences, not whole paragraphs.
    3) If the evidence suggests only direct collection, Article 14 may be relevant conceptually but should be "Not Observed" with 'applicable': true.
    4) Output MUST be valid JSON. Return exactly one JSON object for this article.

    Return only JSON.
    """
    return prompt.strip()

def call_llm_for_structured_result_gemini(prompt: str, response_schema: Any=Result):
    print("Calling llm...GEMINI")
    client = genai.Client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature":0,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            },
    )

    return json.loads(response.text)

def call_llm_for_structured_result_openai(prompt: str, response_schema: Any=Result):
    print("Calling llm...OPENAI")
    client = OpenAI()

    response = client.responses.parse(
        model="gpt-4.1-mini",
        stream=False,
        input=[
            {"role": "system", "content": "You are a GDPR compliance reasoning assistant."},
            {"role": "user", "content": prompt},
        ],
        text_format=response_schema,
        temperature=0,
    )

    return json.loads(response.output_parsed.model_dump_json())
