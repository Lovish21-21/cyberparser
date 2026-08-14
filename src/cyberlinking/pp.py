

import json


def load_linking_data(filepath, entity_type_filter="TECHNIQUE"):
   
    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {line_num}: {e}")
                continue

            if entity_type_filter and rec.get("entity_type") != entity_type_filter:
                continue

            required = ["mention", "label_id", "label", "label_title"]
            if any(rec.get(k) is None for k in required):
                print(f"[WARN] Missing required field at line {line_num}, "
                      f"document={rec.get('document')}")
                continue

            records.append(rec)

    return records


def build_query_text(record):
    
    sentence_left = record.get("sentence_left") or record.get("context_left") or ""
    sentence_right = record.get("sentence_right") or record.get("context_right") or ""
    mention = record.get("mention", "")

    parts = [p for p in [sentence_left.strip(), mention.strip(), sentence_right.strip()] if p]
    return " ".join(parts)


def build_queries(records):
    
    query_texts = []
    gold_label_ids = []
    gold_titles = []

    for rec in records:
        query_texts.append(build_query_text(rec))
        gold_label_ids.append(rec["label_id"])
        gold_titles.append(rec["label_title"])

    return query_texts, gold_label_ids, gold_titles


def build_kb(records):
    
    kb = {}  # label_id -> (description, title, link)

    for rec in records:
        label_id = rec["label_id"]
        if label_id not in kb:
            kb[label_id] = (rec["label"], rec["label_title"], rec.get("label_link", ""))

    kb_ids = list(kb.keys())
    kb_texts = [kb[i][0] for i in kb_ids]
    kb_titles = {i: kb[i][1] for i in kb_ids}
    kb_links = {i: kb[i][2] for i in kb_ids}

    return kb_ids, kb_texts, kb_titles, kb_links