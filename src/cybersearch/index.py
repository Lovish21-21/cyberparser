from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Document, dump_documents_jsonl, load_documents_jsonl
from .preprocess import tokenize


@dataclass(slots=True)
class SearchHit:
    rank: int
    score: float
    document: Document


@dataclass(slots=True)
class SearchIndex:
    documents: list[Document]
    vectorizer: TfidfVectorizer
    matrix: Any

    @classmethod
    def build(cls, documents: list[Document]) -> "SearchIndex":
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=1,
        )
        matrix = vectorizer.fit_transform(document.text for document in documents)
        return cls(documents=documents, vectorizer=vectorizer, matrix=matrix)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1]

        hits: list[SearchHit] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    score=score,
                    document=self.documents[index],
                )
            )
            if len(hits) >= top_k:
                break

        return hits

    def save(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with (artifact_dir / "vectorizer.pkl").open("wb") as handle:
            pickle.dump(self.vectorizer, handle)

        with (artifact_dir / "matrix.pkl").open("wb") as handle:
            pickle.dump(self.matrix, handle)

        dump_documents_jsonl(self.documents, artifact_dir / "documents.jsonl")

    @classmethod
    def load(cls, artifact_dir: Path) -> "SearchIndex":
        with (artifact_dir / "vectorizer.pkl").open("rb") as handle:
            vectorizer = pickle.load(handle)

        with (artifact_dir / "matrix.pkl").open("rb") as handle:
            matrix = pickle.load(handle)

        documents = load_documents_jsonl(artifact_dir / "documents.jsonl")
        return cls(documents=documents, vectorizer=vectorizer, matrix=matrix)
