import json, sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# paths
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

RULES_INPUT      = KB_DIR / "gdpr_rules.json"
RULES_OUTPUT     = KB_DIR / "gdpr_rules_embedding.json"

CHECKLIST_INPUT  = KB_DIR / "gdpr_checklist.json"
CHECKLIST_OUTPUT = KB_DIR / "gdpr_checklist_embedding.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# text builders
def rule_to_text(rule: dict) -> str:
    """Combine a rule's title, intent, triggers, and examples into one
    passage so the embedding captures the full semantic scope."""
    parts = [rule["title"], rule["intent"]]
    parts += rule.get("triggers", [])
    parts += rule.get("examples", [])
    return " ".join(parts)


def clause_to_text(clause: dict) -> str:
    """Combine a clause's description and its checklist questions."""
    parts = [clause["text"]] + clause.get("checklist", [])
    return " ".join(parts)

# main
def main():
    for path in (RULES_INPUT, CHECKLIST_INPUT):
        if not path.exists():
            print(f"ERROR: {path} not found.  Place raw JSON in knowledge_base/.")
            sys.exit(1)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 1. Rules (article-level)
    with open(RULES_INPUT, encoding="utf-8") as f:
        rules = json.load(f)

    for rule in rules:
        text = rule_to_text(rule)
        emb = model.encode(text, convert_to_tensor=True)
        rule["embeddings"] = [e.tolist() for e in emb]

    with open(RULES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)
    print(f"  ✓ {len(rules)} rules → {RULES_OUTPUT.name}")

    # 2. Checklist (clause-level)
    with open(CHECKLIST_INPUT, encoding="utf-8") as f:
        checklist = json.load(f)

    clause_count = 0
    for article in checklist:
        for req in article.get("requirements", []):
            text = clause_to_text(req)
            emb = model.encode(text, convert_to_tensor=True)
            req["embeddings"] = [e.tolist() for e in emb]
            clause_count += 1

    with open(CHECKLIST_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2)
    print(f"  ✓ {clause_count} clauses across {len(checklist)} articles → {CHECKLIST_OUTPUT.name}")

    print("\nDone.  Embedding files are ready for the analyzer.")


if __name__ == "__main__":
    main()
