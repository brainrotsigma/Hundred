"""Utility helpers to load tiny text corpora for Mr Chatter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

_DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "resources" / "sample_corpus.txt"


def load_corpus(path: str | Path | None = None) -> List[str]:
    """Load corpus lines from the provided path or the bundled sample."""
    corpus_path = Path(path) if path is not None else _DEFAULT_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    lines = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def iter_text_fragments(texts: Iterable[str]) -> Iterable[str]:
    """Yield fragments suitable for tokenizer construction."""
    for text in texts:
        yield text
