from __future__ import annotations

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in STOP_WORDS]
