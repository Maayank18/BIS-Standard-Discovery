"""
Chunking Strategy for BIS Standards
=====================================
Two-level chunking for maximum retrieval accuracy:

Level 1 — Standard-level chunk:
    One chunk per IS standard (full summary).
    Best for broad queries about a standard.

Level 2 — Section-level chunk:
    Sub-chunks of individual sections (scope, requirements, etc.).
    Links back to parent standard.
    Best for specific technical queries.

Both levels are indexed together. During retrieval, if a section-level
chunk is retrieved, its parent standard IS number is returned.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.ingestion.pdf_parser import BISStandard

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 2000   # ~512 tokens
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    """A single indexable text chunk."""

    chunk_id: str                  # unique ID
    is_number_full: str            # the canonical IS standard ID
    is_number: str                 # IS number without year
    year: str                      # publication year
    title: str                     # standard title
    category: Optional[str]        # SP21 category
    text: str                      # chunk text (what gets embedded)
    chunk_type: str                # "standard" | "section"
    section_name: Optional[str]    # section name for section-level chunks
    page_start: int                # approximate page number
    keywords: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict:
        """Serialise to dict (for ChromaDB / FAISS metadata store)."""
        return {
            "chunk_id": self.chunk_id,
            "is_number_full": self.is_number_full,
            "is_number": self.is_number,
            "year": self.year,
            "title": self.title,
            "category": self.category or "General",
            "chunk_type": self.chunk_type,
            "section_name": self.section_name or "",
            "page_start": self.page_start,
            "keywords": "|".join(self.keywords),
        }

    def get_embedding_text(self) -> str:
        """
        Text sent to the embedding model.
        Prepend rich context so the embedding captures full semantics.
        """
        parts = [
            f"BIS Standard: {self.is_number_full}",
            f"Title: {self.title}",
        ]
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.section_name:
            parts.append(f"Section: {self.section_name}")
        parts.append(self.text)
        return "\n".join(parts)


def _split_into_sections(text: str) -> List[tuple]:
    """
    Split standard text into (section_name, section_text) pairs.
    Splits on numbered headings like "1. Scope", "2. Requirements", etc.
    """
    # Pattern: number + dot + words + dash (BIS section heading style)
    pattern = re.compile(
        r"(?:^|\n)(\d+(?:\.\d+)*\s+[A-Z][^\n—]*(?:—)?)",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return [("Full Text", text)]

    sections = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip().rstrip("—").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections if sections else [("Full Text", text)]


def create_chunks(standards: List[BISStandard]) -> List[Chunk]:
    """
    Create all chunks from parsed BIS standards.
    Returns both standard-level and section-level chunks.
    """
    all_chunks: List[Chunk] = []
    seen_ids = set()

    for std in standards:
        # ── Level 1: Full standard chunk ──────────────────────────────────
        # Always create one chunk for the entire standard summary
        std_chunk = Chunk(
            chunk_id=f"std_{std.chunk_id}",
            is_number_full=std.is_number_full,
            is_number=std.is_number,
            year=std.year,
            title=std.title,
            category=std.category,
            text=std.full_text[:MAX_CHUNK_CHARS],
            chunk_type="standard",
            section_name=None,
            page_start=std.page_start,
            keywords=std.keywords,
        )
        if std_chunk.chunk_id not in seen_ids:
            all_chunks.append(std_chunk)
            seen_ids.add(std_chunk.chunk_id)

        # ── Level 2: Section-level chunks ─────────────────────────────────
        # Only create sub-chunks if the full text is long enough to warrant it
        if len(std.full_text) > MAX_CHUNK_CHARS:
            sections = _split_into_sections(std.full_text)
            for idx, (sec_name, sec_text) in enumerate(sections):
                if len(sec_text.strip()) < 50:  # skip trivially short sections
                    continue

                # For very long sections, create overlapping sub-chunks
                sub_chunks = _sliding_window(sec_text, MAX_CHUNK_CHARS, OVERLAP_CHARS)

                for sub_idx, sub_text in enumerate(sub_chunks):
                    chunk_id = f"sec_{std.chunk_id}_{idx}_{sub_idx}"
                    if chunk_id in seen_ids:
                        continue
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        is_number_full=std.is_number_full,
                        is_number=std.is_number,
                        year=std.year,
                        title=std.title,
                        category=std.category,
                        text=sub_text,
                        chunk_type="section",
                        section_name=sec_name,
                        page_start=std.page_start,
                        keywords=std.keywords,
                    )
                    all_chunks.append(chunk)
                    seen_ids.add(chunk_id)

    # Also create title-boosted chunks for exact IS number matching
    for std in standards:
        title_chunk_id = f"title_{std.chunk_id}"
        if title_chunk_id in seen_ids:
            continue
        title_text = (
            f"{std.is_number_full} {std.title} "
            + " ".join(std.sections[:5])
            + " "
            + " ".join(std.keywords[:15])
        )
        chunk = Chunk(
            chunk_id=title_chunk_id,
            is_number_full=std.is_number_full,
            is_number=std.is_number,
            year=std.year,
            title=std.title,
            category=std.category,
            text=title_text,
            chunk_type="title",
            section_name=None,
            page_start=std.page_start,
            keywords=std.keywords,
        )
        all_chunks.append(chunk)
        seen_ids.add(title_chunk_id)

    logger.info(
        f"Created {len(all_chunks)} chunks from {len(standards)} standards "
        f"({sum(1 for c in all_chunks if c.chunk_type == 'standard')} standard-level, "
        f"{sum(1 for c in all_chunks if c.chunk_type == 'section')} section-level, "
        f"{sum(1 for c in all_chunks if c.chunk_type == 'title')} title-boosted)"
    )
    return all_chunks


def _sliding_window(text: str, window: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= window:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + window
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > window // 2:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]
