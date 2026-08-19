import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import pypdf


@dataclass
class ParentChunk:
    parent_id: str
    doc_id: str
    filename: str
    content: str
    page_start: int = 1
    page_end: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    filename: str
    content: str
    page_number: int = 1
    contextualized_content: str | None = None
    embedding: list[float] | None = None


class ParentChildChunker:
    """
    Hierarchical chunker that generates parent chunks for rich LLM context
    and child chunks for granular vector & BM25 retrieval.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 150,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 30,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

    def _split_into_words(self, text: str) -> list[str]:
        return text.split()

    def _create_sliding_windows(
        self, words: list[str], window_size: int, overlap: int
    ) -> list[str]:
        if not words:
            return []
        step = max(1, window_size - overlap)
        chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i : i + window_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            if i + window_size >= len(words):
                break
        return chunks

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        filename: str,
        page_start: int = 1,
        page_end: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        metadata = metadata or {}
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        words = self._split_into_words(cleaned_text)

        if not words:
            return [], []

        parent_texts = self._create_sliding_windows(
            words, window_size=self.parent_chunk_size, overlap=self.parent_chunk_overlap
        )

        parent_chunks: list[ParentChunk] = []
        child_chunks: list[ChildChunk] = []

        for p_idx, p_text in enumerate(parent_texts):
            p_id = f"{doc_id}_p_{p_idx}_{uuid.uuid4().hex[:8]}"
            parent_chunk = ParentChunk(
                parent_id=p_id,
                doc_id=doc_id,
                filename=filename,
                content=p_text,
                page_start=page_start,
                page_end=page_end,
                metadata=metadata,
            )
            parent_chunks.append(parent_chunk)

            # Split parent text into granular child chunks
            p_words = self._split_into_words(p_text)
            c_texts = self._create_sliding_windows(
                p_words,
                window_size=self.child_chunk_size,
                overlap=self.child_chunk_overlap,
            )

            for c_idx, c_text in enumerate(c_texts):
                c_id = f"{p_id}_c_{c_idx}"
                child_chunk = ChildChunk(
                    chunk_id=c_id,
                    parent_id=p_id,
                    doc_id=doc_id,
                    filename=filename,
                    content=c_text,
                    page_number=page_start,
                )
                child_chunks.append(child_chunk)

        return parent_chunks, child_chunks

    def chunk_pdf(
        self, pdf_path: str, doc_id: str | None = None
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        reader = pypdf.PdfReader(pdf_path)
        filename = pdf_path.split("/")[-1]
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"

        all_parents: list[ParentChunk] = []
        all_children: list[ChildChunk] = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            parents, children = self.chunk_text(
                text=page_text,
                doc_id=doc_id,
                filename=filename,
                page_start=page_num,
                page_end=page_num,
                metadata={"page": page_num, "total_pages": len(reader.pages)},
            )
            all_parents.extend(parents)
            all_children.extend(children)

        return all_parents, all_children
