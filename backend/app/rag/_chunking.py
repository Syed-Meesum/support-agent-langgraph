"""Split the policy markdown docs into retrievable chunks.

Kept separate from retrieval.py so the chunking rule (and the tests for it) stay
independent of whichever vector store is behind it.
"""

from __future__ import annotations

from pathlib import Path


class Chunk:
    __slots__ = ("content", "position", "source")

    def __init__(self, content: str, source: str, position: int) -> None:
        self.content = content
        self.source = source
        self.position = position


def split_chunks(path: Path) -> list[Chunk]:
    """Split a markdown doc into chunks at '## ' headings."""
    text = path.read_text(encoding="utf-8")
    source = path.name

    chunks: list[Chunk] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            chunks.append(Chunk("\n".join(current).strip(), source, len(chunks)))
            current = []
        current.append(line)
    if current:
        body = "\n".join(current).strip()
        if body:
            chunks.append(Chunk(body, source, len(chunks)))

    return [chunk for chunk in chunks if chunk.content]
