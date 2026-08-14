from __future__ import annotations

from pathlib import Path

from .corpus import load_documents
from .index import SearchHit, SearchIndex


class TfidfSearchEngine:
    def __init__(self, index: SearchIndex) -> None:
        self.index = index

    @classmethod
    def from_text_root(cls, text_root: Path) -> "TfidfSearchEngine":
        documents = load_documents(text_root)
        return cls(SearchIndex.build(documents))

    @classmethod
    def from_artifacts(cls, artifact_dir: Path) -> "TfidfSearchEngine":
        return cls(SearchIndex.load(artifact_dir))

    def save(self, artifact_dir: Path) -> None:
        self.index.save(artifact_dir)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return self.index.search(query, top_k=top_k)
