from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_TEXT_ROOT, DEFAULT_TOP_K
from .engine import TfidfSearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search AnnoCTR documents with TF-IDF cosine ranking."
    )
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--query", type=str, default=None)
    return parser


def load_or_build_engine(text_root: Path, artifact_dir: Path) -> TfidfSearchEngine:
    if (
        artifact_dir / "vectorizer.pkl"
    ).exists() and (artifact_dir / "matrix.pkl").exists() and (artifact_dir / "documents.jsonl").exists():
        return TfidfSearchEngine.from_artifacts(artifact_dir)

    engine = TfidfSearchEngine.from_text_root(text_root)
    engine.save(artifact_dir)
    return engine


def print_results(results) -> None:
    if not results:
        print("No non-zero matches found.")
        return

    for hit in results:
        document = hit.document
        print("=" * 72)
        print(f"Rank        : {hit.rank}")
        print(f"Score       : {hit.score:.4f}")
        print(f"Document ID : {document.doc_id}")
        print(f"Split       : {document.split}")
        print(f"Filename    : {document.filename}")
        print(f"Title       : {document.title}")
        print()
        print(document.text[:500])
        print()


def run() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.text_root.exists():
        raise FileNotFoundError(f"Text root does not exist: {args.text_root}")

    engine = load_or_build_engine(args.text_root, args.artifacts)

    if args.query:
        print_results(engine.search(args.query, top_k=args.top_k))
        return

    while True:
        print()
        query = input("Search query (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        print_results(engine.search(query, top_k=args.top_k))
