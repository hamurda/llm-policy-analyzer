# GDPR scope 
ARTICLES = [5, 6, 12, 13, 14, 15, 44, 45, 46, 47, 48, 49]

# Knowledge-base paths
KB_DIR = "knowledge_base"
GDPR_PARAGRAPHS_JSONL = f"{KB_DIR}/gdpr_paragraphs.jsonl"
GDPR_RULES_EMBEDDING_JSON = f"{KB_DIR}/gdpr_rules_embedding.json"
GDPR_CHECKLIST_EMBEDDING_JSON = f"{KB_DIR}/gdpr_checklist_embedding.json"
GDPR_CHECKLIST_JSON = f"{KB_DIR}/gdpr_checklist.json"
EXAMPLE_POLICY_TXT = f"{KB_DIR}/notion_privacy_policy.txt"

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
OPENAI_MODEL = "gpt-5-mini"

# Retrieval tuning
SIM_THRESHOLD = 0.35
