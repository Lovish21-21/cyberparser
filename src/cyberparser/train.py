import json
import random
import re
from collections import OrderedDict
from itertools import product
from pathlib import Path

import spacy
from spacy.util import minibatch

try:
    from .preprocessing import build_training_examples, collect_entity_labels, ensure_model_directory, load_jsonl
except ImportError:  # pragma: no cover
    from preprocessing import build_training_examples, collect_entity_labels, ensure_model_directory, load_jsonl


def get_default_hyperparameter_grid():
    return [
        {"n_iter": 5, "dropout": 0.1, "batch_size": 4},
        {"n_iter": 5, "dropout": 0.2, "batch_size": 8},
        {"n_iter": 10, "dropout": 0.1, "batch_size": 8},
        {"n_iter": 10, "dropout": 0.2, "batch_size": 8},
        {"n_iter": 20, "dropout": 0.1, "batch_size": 8},
        {"n_iter": 20, "dropout": 0.2, "batch_size": 8},
    ]


def select_best_run(results):
    if not results:
        raise ValueError("No evaluation results were produced.")
    return max(results, key=lambda item: (item["f1"], item["token_acc"]))


def _train_single_config(train_examples, dev_examples, labels, n_iter, dropout, batch_size):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)
    for label in labels:
        ner.add_label(label)

    optimizer = nlp.begin_training()
    for epoch in range(n_iter):
        random.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=batch_size):
            nlp.update(batch, sgd=optimizer, losses=losses, drop=dropout)

    if dev_examples:
        metrics = nlp.evaluate(dev_examples)
        f1 = float(metrics.get("ents_f", 0.0))
        token_acc = float(metrics.get("token_acc", 0.0))
        return nlp, {
            "f1": f1,
            "precision": float(metrics.get("ents_p", 0.0)),
            "recall": float(metrics.get("ents_r", 0.0)),
            "token_acc": token_acc,
            "n_iter": n_iter,
            "dropout": dropout,
            "batch_size": batch_size,
        }

    return nlp, {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "token_acc": 0.0,
        "n_iter": n_iter,
        "dropout": dropout,
        "batch_size": batch_size,
    }


def train_spacy_ner(train_path, dev_path=None, output_dir="models/cti_spacy_ner", n_iter=20, dropout=0.2, batch_size=8, hyperparameter_grid=None):
    train_records = load_jsonl(train_path)
    labels = collect_entity_labels(train_records)
    print(f"[INFO] Loaded {len(train_records)} training records")
    print(f"[INFO] Labels: {labels}")

    nlp = spacy.blank("en")
    train_examples = build_training_examples(train_records, nlp)
    print(f"[INFO] Generated {len(train_examples)} training examples")

    if dev_path:
        dev_records = load_jsonl(dev_path)
        dev_examples = build_training_examples(dev_records, nlp)
        print(f"[INFO] Loaded {len(dev_examples)} dev examples")
    else:
        dev_examples = train_examples[: min(500, len(train_examples))]

    if hyperparameter_grid is None:
        grid = get_default_hyperparameter_grid()
    else:
        grid = hyperparameter_grid

    results = []
    best_model = None
    best_metadata = None

    for config in grid:
        cfg_n_iter = int(config.get("n_iter", n_iter))
        cfg_dropout = float(config.get("dropout", dropout))
        cfg_batch_size = int(config.get("batch_size", batch_size))

        candidate_model, candidate_meta = _train_single_config(
            train_examples=train_examples,
            dev_examples=dev_examples,
            labels=labels,
            n_iter=cfg_n_iter,
            dropout=cfg_dropout,
            batch_size=cfg_batch_size,
        )
        candidate_meta["config"] = config
        print(
            f"[INFO] Candidate config: n_iter={cfg_n_iter}, dropout={cfg_dropout}, batch_size={cfg_batch_size}, "
            f"f1={candidate_meta['f1']:.4f}, token_acc={candidate_meta['token_acc']:.4f}"
        )
        results.append(candidate_meta)

        if best_model is None or (candidate_meta["f1"], candidate_meta["token_acc"]) > (best_metadata["f1"], best_metadata["token_acc"]):
            best_model = candidate_model
            best_metadata = candidate_meta

    if best_model is None:
        raise RuntimeError("No valid model could be trained.")

    if output_dir:
        ensure_model_directory(output_dir)
        best_model.to_disk(output_dir)
        print(f"[INFO] Saved best spaCy model to {output_dir}")

    best_metadata["labels"] = labels
    best_metadata["train_examples"] = len(train_examples)
    best_metadata["dev_examples"] = len(dev_examples)
    best_metadata["search_results"] = results
    best_metadata["best_config"] = best_metadata["config"]
    best_metadata.pop("config", None)

    return best_model, best_metadata


def normalize_entity_key(label):
    if label is None:
        return None
    key = str(label).strip()
    if not key or key == "O":
        return None
    key = key.replace("/", "_").replace("-", "_").replace(" ", "_")
    key = re.sub(r"[^A-Za-z0-9_]", "_", key)
    return key.strip("_").lower()


def parse_entities_to_dict(entities):
    grouped = OrderedDict()
    for text, label in entities:
        if text is None:
            continue
        value = str(text).strip()
        if not value:
            continue
        key = normalize_entity_key(label)
        if not key:
            continue
        grouped.setdefault(key, [])
        if value not in grouped[key]:
            grouped[key].append(value)
    return dict(grouped)


def parse_text(model, text):
    doc = model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def parse_text_to_dict(model, text):
    return parse_entities_to_dict(parse_text(model, text))


def parse_file_to_dict(model, path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_text_to_dict(model, text)
