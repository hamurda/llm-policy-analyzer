import json
from typing import List, Dict, Any


def load_from_jsonlines(path: str) -> List[Dict[str, Any]]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def load_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def write_to_json(path:str, data:Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data,f, indent=2)

def filter_by_article(rules:List[Dict[str, Any]], article_list: List[int]):
    article_set = {str(a) for a in article_list}
    
    return [
        r for r in rules
        if str(r.get("article")) in article_set
        or str(r.get("structure", {}).get("article")) in article_set
    ]