# Scope
ARTICLES = [5, 6, 12, 13, 14, 15, 44, 45, 46, 47, 48, 49]

# Knowledge Base
GDPR_PARAGRAPHS_JSONL = "knowledge_base/legal/gdpr/gdpr_paragraphs.jsonl"
GDPR_RULES_EMBEDDING_JSON = "knowledge_base/legal/gdpr/gdpr_rules_embedding.json"
GDPR_CHECKLIST_EMBEDDING_JSON = "knowledge_base/legal/gdpr/gdpr_checklist_embedding.json"
GDPR_CHECKLIST_JSON = "knowledge_base/legal/gdpr/gdpr_checklist.json"
NOTION_POLICY_JSONL= "knowledge_base/policy/notion/notion_privacy.jsonl"

# Embedding
DEVICE = "cpu"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GDPR_DB_NAME = "gdpr_db"
POLICY_DB_NAME = "notion_db"

# LLM models
OPENAI_MODEL = "gpt-4.1-mini"
GEMINI_MODEL = "gemini-2.5-flash"

# Retrieval
TOP_K_PER_ARTICLE = 6
SIM_THRESHOLD = 0.35  # semantic similarity threshold (tune)

# Result file
EVAL_RESULT = "out/gdpr_policy_eval.json"