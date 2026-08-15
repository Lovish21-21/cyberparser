import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import spacy
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Example


def split_bio_label(label: str) -> Tuple[str, str]:
    if label is None or label == "":
        return "O", "O"

    value = str(label).strip()
    if value in {"O", "-", ""}:
        return "O", "O"

    if value.startswith("B-"):
        return "B", value[2:]
    if value.startswith("I-"):
        return "I", value[2:]
    return "B", value


def normalize_label(label: str) -> str:
    _, entity_label = split_bio_label(label)
    return entity_label


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


def _annotate_record_to_entities(record: dict, doc: Doc) -> List[Tuple[int, int, str]]:
    text = record.get("text") or " ".join(record.get("tokens", []))
    tokens = record.get("tokens") or []
    raw_labels = record.get("all_tags") or record.get("labels") or record.get("all_labels") or []

    if not tokens or len(tokens) != len(raw_labels):
        return []

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

        prefix, normalized = split_bio_label(label)
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
        elif prefix == "I" and normalized == current_label:
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

    return filtered


def _record_to_doc_with_ents(record: dict, nlp: spacy.Language) -> Optional[Doc]:
    """Build a Doc directly from the record's pre-tokenized `tokens` list and
    assign entity spans by TOKEN INDEX (not character search).

    This is the key fix: previously entities were located by re-finding token
    text inside a re-tokenized `nlp.make_doc(text)`, which frequently produced
    character offsets that didn't line up with spaCy's own tokenization
    (punctuation, markdown links, hyphenation, etc). Any span that didn't sit
    exactly on a token boundary was silently DROPPED by spaCy during training
    (the "W030 misaligned entities" warnings) -- losing the majority of gold
    entities and leaving the NER model with almost nothing to learn from,
    which is why inference returns empty entities.

    Since the source data already gives us the correct token boundaries, we
    build the Doc directly from `words=tokens` so there is no re-tokenization
    step and therefore no possible misalignment.
    """
    tokens = record.get("tokens") or []
    raw_labels = record.get("all_tags") or record.get("labels") or record.get("all_labels") or []

    if not tokens or len(tokens) != len(raw_labels):
        return None

    words = [str(t) if t is not None and str(t).strip() != "" else " " for t in tokens]
    doc = Doc(nlp.vocab, words=words)

    spans = []
    current_label = None
    start_i = None

    for i, label in enumerate(raw_labels):
        prefix, normalized = split_bio_label(label)

        if normalized == "O":
            if current_label is not None:
                spans.append(Span(doc, start_i, i, label=current_label))
                current_label = None
                start_i = None
            continue

        if current_label is None:
            current_label = normalized
            start_i = i
        elif prefix == "I" and normalized == current_label:
            continue  # extend current span
        else:
            spans.append(Span(doc, start_i, i, label=current_label))
            current_label = normalized
            start_i = i

    if current_label is not None:
        spans.append(Span(doc, start_i, len(raw_labels), label=current_label))

    doc.ents = tuple(spans)
    return doc


def build_training_examples(records: Iterable[dict], nlp: spacy.Language):
    examples = []
    for record in records:
        doc = _record_to_doc_with_ents(record, nlp)
        if doc is None:
            continue
        examples.append(Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}))
    return examples


def build_docbin_documents(records: Iterable[dict], nlp: spacy.Language):
    docs = []
    for record in records:
        doc = _record_to_doc_with_ents(record, nlp)
        if doc is None:
            continue
        docs.append(doc)
    return docs


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


def prepare_spacy_corpus(train_path: str, dev_path: Optional[str] = None, output_dir: str = "models/cti_spacy_ner/corpus"):
    ensure_model_directory(output_dir)
    nlp = spacy.blank("en")

    train_records = load_jsonl(train_path)
    train_docs = build_docbin_documents(train_records, nlp)
    train_out = Path(output_dir) / "train.spacy"
    DocBin(docs=train_docs).to_disk(str(train_out))

    dev_records = None
    if dev_path:
        dev_records = load_jsonl(dev_path)
    elif len(train_records) > 1:
        split_index = max(1, int(len(train_records) * 0.2))
        dev_records = train_records[-split_index:]
        train_records = train_records[:-split_index] if split_index < len(train_records) else []
        train_docs = build_docbin_documents(train_records, nlp)
        DocBin(docs=train_docs).to_disk(str(train_out))

    dev_out = None
    if dev_records is not None:
        dev_docs = build_docbin_documents(dev_records, nlp)
        dev_out = Path(output_dir) / "dev.spacy"
        DocBin(docs=dev_docs).to_disk(str(dev_out))

    return {
        "train_count": len(train_docs),
        "dev_count": len(dev_docs) if dev_records is not None else 0,
        "train_path": str(train_out),
        "dev_path": str(dev_out) if dev_out is not None else None,
    }


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