import json
import sys
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cyberparser.preprocessing import _annotate_record_to_entities, prepare_spacy_corpus
from src.cyberparser.train import parse_entities_to_dict


def test_parse_entities_to_dict_groups_labels():
    ents = [
        ("LockBit", "GROUP"),
        ("2023", "DATE"),
        ("Cobalt Strike", "TOOL"),
        ("Europe", "LOC"),
        ("LockBit", "GROUP"),
    ]

    result = parse_entities_to_dict(ents)

    assert result["group"] == ["LockBit"]
    assert result["date"] == ["2023"]
    assert result["tool"] == ["Cobalt Strike"]
    assert result["loc"] == ["Europe"]


def test_prepare_spacy_corpus_creates_docbin_files(tmp_path):
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"

    train_path.write_text(
        json.dumps({
            "text": "LockBit attacked Europe.",
            "tokens": ["LockBit", "attacked", "Europe", "."],
            "all_tags": ["GROUP", "O", "LOC", "O"],
        }) + "\n",
        encoding="utf-8",
    )
    dev_path.write_text(
        json.dumps({
            "text": "Cobalt Strike was used.",
            "tokens": ["Cobalt", "Strike", "was", "used", "."],
            "all_tags": ["TOOL", "TOOL", "O", "O", "O"],
        }) + "\n",
        encoding="utf-8",
    )

    result = prepare_spacy_corpus(str(train_path), str(dev_path), str(tmp_path / "corpus"))

    assert result["train_count"] == 1
    assert result["dev_count"] == 1
    assert Path(result["train_path"]).exists()
    assert Path(result["dev_path"]).exists()


def test_annotate_record_keeps_adjacent_b_tags_separate():
    nlp = spacy.blank("en")
    record = {
        "text": "APT28 APT29 campaign",
        "tokens": ["APT28", "APT29", "campaign"],
        "all_tags": ["B-GROUP", "B-GROUP", "O"],
    }

    doc = nlp.make_doc(record["text"])
    entities = _annotate_record_to_entities(record, doc)

    assert entities == [(0, 5, "GROUP"), (6, 11, "GROUP")]
