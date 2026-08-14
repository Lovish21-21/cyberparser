from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json


@dataclass(slots=True)
class Document:
    doc_id: str
    split: str
    filename: str
    relative_path: str
    title: str
    text: str


def load_documents(text_root: Path) -> list[Document]:
    documents: list[Document] = []

    for split_dir in sorted(path for path in text_root.iterdir() if path.is_dir()):
        for file_path in sorted(split_dir.rglob("*.txt")):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
            title = non_empty_lines[0] if non_empty_lines else file_path.stem

            documents.append(
                Document(
                    doc_id=file_path.stem,
                    split=split_dir.name,
                    filename=file_path.name,
                    relative_path=str(file_path.relative_to(text_root)).replace("\\", "/"),
                    title=title,
                    text=text,
                )
            )

    return documents


def dump_documents_jsonl(documents: Iterable[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False))
            handle.write("\n")


def load_documents_jsonl(input_path: Path) -> list[Document]:
    documents: list[Document] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            documents.append(Document(**payload))
    return documents
