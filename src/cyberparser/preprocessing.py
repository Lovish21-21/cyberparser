import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import spacy
from spacy.tokens import Doc
from spacy.training import Example


def normalize_label(label: str) -> str:
    if label is None or label == "":
        return "O"
    value = str(label).strip()
    if value in {"O", "-", ""}:
        return "O"
    if value.startswith("B-") or value.startswith("I-"):
        value = value[2:]
    return value


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _find_token_spans(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    spans = []
    cursor = 0
    for token in tokens:
        if token is None:
            spans.append((cursor, cursor))
            continue

        token_value = str(token)
        if token_value == "":
            spans.append((cursor, cursor))
            continue

        idx = text.find(token_value, cursor)
        if idx == -1:
            remainder = text[cursor:]
            match = re.search(re.escape(token_value), remainder)
            if match:
                idx = cursor + match.start()
            else:
                idx = cursor

        if idx < cursor:
            idx = cursor

        start = idx
        end = idx + len(token_value)
        spans.append((start, end))
        cursor = max(end, cursor)
    return spans


def build_training_examples(records: Iterable[dict], nlp: spacy.Language):
    examples = []
    for record in records:
        text = record.get("text") or " ".join(record.get("tokens", []))
        tokens = record.get("tokens") or []
        raw_labels = record.get("all_tags") or record.get("labels") or record.get("all_labels") or []

        if not tokens or len(tokens) != len(raw_labels):
            continue

        doc = nlp.make_doc(text)
        spans = _find_token_spans(text, tokens)

        entities = []
        current_label = None
        start = end = None

        for token, label, span in zip(tokens, raw_labels, spans):
            token_value = str(token).strip()
            if not token_value:
                if current_label is not None:
                    entities.append((start, end, current_label))
                    current_label = None
                    start = end = None
                continue

            normalized = normalize_label(label)
            if normalized == "O":
                if current_label is not None:
                    entities.append((start, end, current_label))
                    current_label = None
                    start = end = None
                continue

            span_start, span_end = span
            if span_start < 0 or span_end <= span_start:
                if current_label is not None:
                    entities.append((start, end, current_label))
                    current_label = None
                    start = end = None
                continue

            if span_end > len(doc.text):
                span_end = len(doc.text)

            if current_label is None:
                current_label = normalized
                start, end = span_start, span_end
            elif normalized == current_label:
                end = max(end, span_end)
            else:
                entities.append((start, end, current_label))
                current_label = normalized
                start, end = span_start, span_end

        if current_label is not None:
            entities.append((start, end, current_label))

        filtered = []
        for s, e, label in entities:
            if s is None or e is None:
                continue
            if s < 0 or e <= s or e > len(doc.text):
                continue
            span_text = doc.text[s:e]
            if not span_text.strip():
                continue
            if re.fullmatch(r"[\W_]+", span_text.strip()):
                continue
            filtered.append((s, e, label))

        examples.append(Example.from_dict(doc, {"entities": filtered}))

    return examples


def collect_entity_labels(records: Iterable[dict]) -> List[str]:
    labels = set()
    for record in records:
        raw_labels = record.get("all_tags") or record.get("labels") or record.get("all_labels") or []
        for value in raw_labels:
            label = normalize_label(value)
            if label != "O":
                labels.add(label)
    return sorted(labels)


def ensure_model_directory(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    nlp = spacy.blank("en")
    sample = load_jsonl(r"E:\CTI_file _parser\dataset\train_ext.json")[:2]
    examples = build_training_examples(sample, nlp)
    print(f"Generated examples: {len(examples)}")
    for ex in examples[:1]:
        print(ex.reference.text)
        print(ex.reference.ents)


def _get_label_field(record):
    for key in ("all_tags", "labels", "all_labels"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def load_data(filepath):
    all_tokens = []
    all_labels = []
    all_ids = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            tokens = record.get("tokens")
            labels = _get_label_field(record)

            if tokens is None or labels is None:
                print(f"[WARN] Missing tokens/labels at line {line_num}, id={record.get('id')}")
                continue

            if len(tokens) != len(labels):
                print(f"[WARN] Length mismatch at line {line_num}, id={record.get('id')}: "
                      f"{len(tokens)} tokens vs {len(labels)} labels")
                continue

            all_tokens.append(tokens)
            all_labels.append(labels)
            all_ids.append(record.get("id"))

    return all_tokens, all_labels, all_ids
