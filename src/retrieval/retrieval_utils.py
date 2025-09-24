from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import EMBEDDING_MODEL, DEVICE, GDPR_DB_NAME, POLICY_DB_NAME, ARTICLES, GDPR_PARAGRAPHS_JSONL, GDPR_RULES_EMBEDDING_JSON, GDPR_CHECKLIST_EMBEDDING_JSON, NOTION_POLICY_JSONL
from src.utils.helpers import load_from_json,load_from_jsonlines, filter_by_article


#Embedding for Chroma
langchain_embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': DEVICE}
)

#Embedding for cos_sim
st_embedding = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

#DBs
gdpr_db = Chroma(persist_directory=GDPR_DB_NAME, embedding_function=langchain_embedding)
policy_db = Chroma(persist_directory=POLICY_DB_NAME, embedding_function=langchain_embedding)

# Rules and chunks
_all_gdpr_rules = load_from_json(GDPR_RULES_EMBEDDING_JSON)
gdpr_rules = filter_by_article(_all_gdpr_rules, ARTICLES)

_all_gdpr_paras = load_from_jsonlines(GDPR_PARAGRAPHS_JSONL)
gdpr_paras = filter_by_article(_all_gdpr_paras, ARTICLES)

all_checklists = load_from_json(GDPR_CHECKLIST_EMBEDDING_JSON)
policy_chunks = load_from_jsonlines(NOTION_POLICY_JSONL)


# Get law paragraphs by article
def get_gdpr_paragraphs_by_article(article:int) -> list[dict]:
    return [
        a for a in gdpr_paras
        if a.get("structure", {}).get("article") == article
    ]