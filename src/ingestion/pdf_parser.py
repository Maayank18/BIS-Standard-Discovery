"""
BIS SP 21 PDF Parser
====================
Parses the 929-page BIS SP 21 document into structured standard summaries.

Each standard page has the format:
    SP 21 : 2005                        ← header
    SUMMARY OF
    IS XXXX : YYYY [TITLE]              ← standard identifier + title
    (First/Second/... Revision)         ← optional revision marker
    
    1. Scope — ...
    2. ...
    
    For detailed information, refer to IS XXXX:YYYY ...  ← footer

The parser extracts one document per IS standard with rich metadata.
"""
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Regex patterns ──────────────────────────────────────────────────────────
# Matches: IS 269 : 1989, IS 2185 (Part 2) : 1983, IS 1489 (Part 1) : 1991
IS_NUMBER_PATTERN = re.compile(
    r"IS\s+(\d+(?:\s*\(Part\s*\d+\))?(?:\s*\(Sec\s*\d+\))?)\s*:\s*(\d{4})",
    re.IGNORECASE,
)

SUMMARY_OF_PATTERN = re.compile(r"SUMMARY\s+OF", re.IGNORECASE)

# Section heading: "1. Scope", "2. Requirements", etc.
SECTION_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z][a-zA-Z\s]+—", re.MULTILINE)

# SP21 page header pattern to strip
SP_HEADER_PATTERN = re.compile(r"SP\s*21\s*:\s*2005", re.IGNORECASE)

# BIS section categories (from contents page)
SECTION_CATEGORIES = {
    1: "Cement and Concrete",
    2: "Building Limes",
    3: "Stones",
    4: "Wood Products for Building",
    5: "Gypsum Building Materials",
    6: "Timber",
    7: "Bitumen and Tar Products",
    8: "Floor, Wall, Roof Coverings and Finishes",
    9: "Water Proofing and Damp Proofing Materials",
    10: "Sanitary Appliances and Water Fittings",
    11: "Builder's Hardware",
    12: "Wood Products",
    13: "Doors, Windows and Shutters",
    14: "Concrete Reinforcement",
    15: "Structural Steels",
    16: "Light Metal and Their Alloys",
    17: "Structural Shapes",
    18: "Welding Electrodes and Wires",
    19: "Threaded Fasteners and Rivets",
    20: "Wire Ropes and Wire Products",
    21: "Glass",
    22: "Fillers, Stoppers and Putties",
    23: "Thermal Insulation Materials",
    24: "Plastics",
    25: "Conductors and Cables",
    26: "Wiring Accessories",
    27: "General",
}


@dataclass
class BISStandard:
    """Represents a single BIS standard summary extracted from SP 21."""

    is_number: str           # e.g., "IS 269"
    is_number_full: str      # e.g., "IS 269 : 1989"
    year: str                # e.g., "1989"
    title: str               # e.g., "ORDINARY PORTLAND CEMENT, 33 GRADE"
    revision: Optional[str]  # e.g., "Fifth Revision"
    full_text: str           # complete summary text
    sections: List[str]      # list of section headings found
    sp21_section: Optional[int]  # which SP21 section number
    category: Optional[str]  # derived from sp21_section
    page_start: int          # approximate page number
    keywords: List[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        """Unique identifier for this standard."""
        return self.is_number_full.replace(" ", "_").replace(":", "").replace("/", "_")

    @property
    def short_id(self) -> str:
        """Clean IS number for display."""
        return self.is_number_full


def _clean_text(text: str) -> str:
    """Remove PDF artifacts and normalize whitespace."""
    # Remove page headers like "SP 21 : 2005"
    text = SP_HEADER_PATTERN.sub("", text)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


def _extract_is_number(text: str) -> Optional[re.Match]:
    """Find the first IS standard number in text."""
    return IS_NUMBER_PATTERN.search(text)


def _extract_revision(text: str) -> Optional[str]:
    """Extract revision marker like 'First Revision', 'Second Revision'."""
    rev_pattern = re.compile(
        r"\((First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\s+Revision\)",
        re.IGNORECASE,
    )
    match = rev_pattern.search(text)
    return match.group(0).strip("()") if match else None


def _extract_title(text: str, is_match: re.Match) -> str:
    """
    Extract standard title. Title is on same line as IS number or
    the line(s) immediately following.
    """
    # Get text after IS number
    after_is = text[is_match.end():].strip()
    # Title is typically the first line after IS number
    lines = after_is.split("\n")
    title_parts = []
    for line in lines[:3]:
        line = line.strip()
        # Stop if we hit a revision marker or numbered section
        if re.match(r"\((First|Second|Third|Fourth|Fifth)", line, re.IGNORECASE):
            break
        if re.match(r"^\d+\.", line):
            break
        if line:
            title_parts.append(line)
    return " ".join(title_parts).strip()


def _extract_sections(text: str) -> List[str]:
    """Extract section headings from the standard summary."""
    sections = []
    for match in SECTION_PATTERN.finditer(text):
        heading = match.group(0).strip()
        # Clean up the heading
        heading = re.sub(r"\s+", " ", heading).rstrip("—").strip()
        sections.append(heading)
    return sections[:20]  # cap at 20 sections


def _extract_keywords(standard: "BISStandard") -> List[str]:
    """
    Extract domain keywords for boosted BM25 matching.
    Combines title words + section headings.
    """
    text = f"{standard.title} {standard.is_number_full} {' '.join(standard.sections)}"
    # Extract significant words (4+ chars, not stopwords)
    stopwords = {
        "with", "from", "that", "this", "shall", "should", "where",
        "which", "their", "have", "been", "also", "each", "such",
        "than", "then", "when", "will", "used", "uses", "use",
        "part", "section", "standard", "indian", "requirements",
    }
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    keywords = [w for w in set(words) if w not in stopwords]
    return keywords[:30]


class BISPDFParser:
    """
    Parses BIS SP 21 PDF into structured BISStandard objects.

    Strategy:
        1. Extract all text page by page using PyMuPDF
        2. Split on "SUMMARY OF" markers to isolate each standard
        3. Parse IS number, title, revision, and sections from each block
        4. Assign SP21 section/category based on page range heuristics
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)

    def parse(self) -> List[BISStandard]:
        """Parse entire PDF and return list of BISStandard objects."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        logger.info(f"Opening PDF: {self.pdf_path} ...")
        doc = fitz.open(str(self.pdf_path))
        total_pages = len(doc)
        logger.info(f"Total pages: {total_pages}")

        standards: List[BISStandard] = []
        current_block: List[str] = []
        current_page: int = 0
        in_summary = False

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            text = _clean_text(text)

            if not text.strip():
                continue

            if SUMMARY_OF_PATTERN.search(text):
                # Save previous block if it exists
                if current_block and in_summary:
                    block_text = "\n".join(current_block)
                    std = self._parse_block(block_text, current_page)
                    if std:
                        standards.append(std)

                # Start new block
                current_block = [text]
                current_page = page_num + 1
                in_summary = True
            elif in_summary:
                current_block.append(text)

        # Don't forget the last block
        if current_block and in_summary:
            block_text = "\n".join(current_block)
            std = self._parse_block(block_text, current_page)
            if std:
                standards.append(std)

        doc.close()

        # Post-process: assign categories based on IS number ranges
        self._assign_categories(standards)

        logger.info(f"Parsed {len(standards)} BIS standards from PDF")
        return standards

    def _parse_block(self, text: str, page_num: int) -> Optional[BISStandard]:
        """Parse a single standard block into a BISStandard."""
        is_match = _extract_is_number(text)
        if not is_match:
            return None

        # Build the full IS number string
        number_part = is_match.group(1).strip()
        # Normalize part notation
        number_part = re.sub(r"\s*\(\s*Part\s*", " (Part ", number_part)
        year = is_match.group(2).strip()
        is_number = f"IS {number_part}"
        is_number_full = f"IS {number_part} : {year}"

        title = _extract_title(text, is_match)
        revision = _extract_revision(text)

        sections = _extract_sections(text)

        std = BISStandard(
            is_number=is_number,
            is_number_full=is_number_full,
            year=year,
            title=title,
            revision=revision,
            full_text=text.strip(),
            sections=sections,
            sp21_section=None,
            category=None,
            page_start=page_num,
        )
        std.keywords = _extract_keywords(std)
        return std

    def _assign_categories(self, standards: List[BISStandard]) -> None:
        """
        Assign SP21 section/category based on IS number and known ranges.
        This uses knowledge of the SP21 contents table structure.
        """
        # IS number ranges by category (approximate, based on SP21 structure)
        # Section 1: Cement and Concrete — IS numbers commonly in these ranges
        cement_concrete_numbers = {
            "269", "455", "456", "458", "459", "516", "1343", "1489",
            "2116", "2185", "2386", "3466", "4031", "4926", "6452",
            "6909", "7861", "8041", "8042", "8043", "8112", "12269",
            "12330", "12600", "383", "432", "1786", "2062",
        }
        steel_numbers = {"432", "1786", "2062", "1977", "2830"}

        for std in standards:
            # Extract base IS number (without Part)
            base_num = re.search(r"IS\s+(\d+)", std.is_number)
            if not base_num:
                continue
            num_str = base_num.group(1)

            title_lower = std.title.lower()

            # Category assignment by content keywords
            if any(
                kw in title_lower
                for kw in ["cement", "concrete", "aggregate", "mortar", "grout", "coping"]
            ):
                std.sp21_section = 1
                std.category = "Cement and Concrete"
            elif any(kw in title_lower for kw in ["lime", "limestone"]):
                std.sp21_section = 2
                std.category = "Building Limes"
            elif any(kw in title_lower for kw in ["stone", "granite", "marble", "slate"]):
                std.sp21_section = 3
                std.category = "Stones"
            elif any(kw in title_lower for kw in ["steel", "reinforcement", "bar", "wire rope"]):
                std.sp21_section = 14
                std.category = "Concrete Reinforcement"
            elif any(kw in title_lower for kw in ["structural steel", "i-section", "channel"]):
                std.sp21_section = 15
                std.category = "Structural Steels"
            elif any(kw in title_lower for kw in ["glass"]):
                std.sp21_section = 21
                std.category = "Glass"
            elif any(kw in title_lower for kw in ["timber", "wood", "plywood", "particle board"]):
                std.sp21_section = 6
                std.category = "Timber"
            elif any(kw in title_lower for kw in ["bitumen", "tar", "asphalt"]):
                std.sp21_section = 7
                std.category = "Bitumen and Tar Products"
            elif any(kw in title_lower for kw in ["tile", "floor", "roof", "wall covering", "terrazzo"]):
                std.sp21_section = 8
                std.category = "Floor, Wall, Roof Coverings and Finishes"
            elif any(kw in title_lower for kw in ["waterproof", "damp proof"]):
                std.sp21_section = 9
                std.category = "Water Proofing and Damp Proofing Materials"
            elif any(kw in title_lower for kw in ["pipe", "fitting", "valve", "tap", "cistern"]):
                std.sp21_section = 10
                std.category = "Sanitary Appliances and Water Fittings"
            elif any(kw in title_lower for kw in ["door", "window", "shutter"]):
                std.sp21_section = 13
                std.category = "Doors, Windows and Shutters"
            elif any(kw in title_lower for kw in ["gypsum", "plaster of paris"]):
                std.sp21_section = 5
                std.category = "Gypsum Building Materials"
            elif any(kw in title_lower for kw in ["thermal insulation", "insulating"]):
                std.sp21_section = 23
                std.category = "Thermal Insulation Materials"
            elif any(kw in title_lower for kw in ["plastic", "pvc", "polyethylene"]):
                std.sp21_section = 24
                std.category = "Plastics"
            else:
                std.sp21_section = 27
                std.category = "General"
