import argparse
import json

import spacy

try:
    from .preprocessing import ensure_model_directory, load_jsonl
    from .train import parse_file_to_dict, parse_text, parse_text_to_dict, train_spacy_ner
except ImportError:  # pragma: no cover
    from preprocessing import ensure_model_directory, load_jsonl
    from train import parse_file_to_dict, parse_text, parse_text_to_dict, train_spacy_ner


def main():
    parser = argparse.ArgumentParser(description="CTI report parser using spaCy NER")
    parser.add_argument("--train", required=False, default=None, help="Path to JSONL training data")
    parser.add_argument("--dev", required=False, help="Optional JSONL dev/eval data")
    parser.add_argument("--output", default="models/cti_spacy_ner", help="Directory to save trained model")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to an already-trained spaCy model directory. If set, skips training entirely "
        "and loads this model for parsing instead.",
    )
    parser.add_argument("--text", default=None, help="Text snippet to parse")
    parser.add_argument("--text-file", default=None, help="Path to a text file to parse")
    parser.add_argument("--epochs", type=int, default=20, help="Number of spaCy training epochs")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout used during spaCy NER training")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for spaCy NER training")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only use the first N training records (and first N dev records). "
        "Useful for a fast sanity-check run before committing to the full dataset.",
    )

    args = parser.parse_args()

    if args.model_path:
        print(f"[INFO] Loading existing spaCy model from {args.model_path} (skipping training)...")
        model = spacy.load(args.model_path)
        metadata = None
    else:
        if not args.train:
            parser.error("--train is required unless --model-path is provided")

        print("[INFO] Training spaCy CTI NER model...")
        model, metadata = train_spacy_ner(
            train_path=args.train,
            dev_path=args.dev,
            output_dir=args.output,
            n_iter=args.epochs,
            dropout=args.dropout,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(f"[INFO] Model metadata: {metadata}")

    if args.text:
        print("\n[INFO] Parsing provided text:")
        parsed = parse_text_to_dict(model, args.text)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

    if args.text_file:
        print("\n[INFO] Parsing text file:")
        parsed = parse_file_to_dict(model, args.text_file)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

    return model


if __name__ == "__main__":
    main()