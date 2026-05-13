"""
Document Processor — robust PDF ingestion + section-aware table extraction.

Improvements:
- Stable section heading tracking
- Prevents OCR/table rows becoming headings
- Cleaner table extraction
- Better pharma document compatibility
- Safer chunk generation
"""

import pdfplumber
import re

from dataclasses import dataclass, field


# ------------------------------------------------------------------ #
# Chunk Model
# ------------------------------------------------------------------ #

@dataclass
class DocumentChunk:
    """
    Represents a single extracted chunk.

    chunk_type:
        - text
        - table
    """

    chunk_id: str

    page_number: int

    section_heading: str

    chunk_type: str

    content: str

    table_data: list[list] = field(
        default_factory=list
    )


# ------------------------------------------------------------------ #
# Document Processor
# ------------------------------------------------------------------ #

class DocumentProcessor:

    """
    Extracts:
    - text blocks
    - structured tables
    - section-aware chunks
    """

    # stricter heading pattern
    HEADING_PATTERN = re.compile(
        r"^\d+(\.\d+)*\s+[A-Z][A-Za-z\s,\-&()]{5,}$"
    )

    def __init__(
        self,
        pdf_path: str,
    ):

        self.pdf_path = pdf_path

        self.chunks: list[DocumentChunk] = []

        self._current_heading = "General"

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def process(self) -> list[DocumentChunk]:

        """
        Main pipeline entry point.
        """

        self.chunks = []

        with pdfplumber.open(self.pdf_path) as pdf:

            for page_num, page in enumerate(
                pdf.pages,
                start=1,
            ):

                self._process_page(
                    page,
                    page_num,
                )

        print(
            f"[DocumentProcessor] "
            f"Extracted {len(self.chunks)} chunks "
            f"from {self.pdf_path}"
        )

        return self.chunks

    def get_text_chunks(
        self,
    ) -> list[DocumentChunk]:

        return [
            c for c in self.chunks
            if c.chunk_type == "text"
        ]

    def get_table_chunks(
        self,
    ) -> list[DocumentChunk]:

        return [
            c for c in self.chunks
            if c.chunk_type == "table"
        ]

    def get_full_text(
        self,
    ) -> str:

        return "\n\n".join(
            c.content
            for c in self.chunks
        )

    # ------------------------------------------------------------------ #
    # Page Processing
    # ------------------------------------------------------------------ #

    def _process_page(
        self,
        page,
        page_num: int,
    ):

        """
        Extracts:
        - page text
        - tables
        """

        # ------------------------------------------------------------ #
        # Step 1 — Extract raw page text
        # ------------------------------------------------------------ #

        raw_text = page.extract_text()

        if raw_text:

            self._extract_headings_from_text(
                raw_text
            )

            text_chunk = DocumentChunk(
                chunk_id=f"p{page_num}_text",
                page_number=page_num,
                section_heading=self._current_heading,
                chunk_type="text",
                content=raw_text.strip(),
            )

            self.chunks.append(text_chunk)

        # ------------------------------------------------------------ #
        # Step 2 — Extract tables
        # ------------------------------------------------------------ #

        try:

            tables = page.extract_tables()

        except Exception as e:

            print(
                f"[DocumentProcessor] "
                f"Table extraction failed on "
                f"page {page_num}: {e}"
            )

            tables = []

        for idx, table in enumerate(tables):

            if not table:
                continue

            clean_table = self._clean_table(
                table
            )

            if not clean_table:
                continue

            content_str = self._table_to_text(
                clean_table
            )

            # IMPORTANT:
            # DO NOT update headings from table content.
            # This caused OCR/table rows to become
            # fake section headings.

            chunk = DocumentChunk(
                chunk_id=f"p{page_num}_table{idx}",
                page_number=page_num,
                section_heading=self._current_heading,
                chunk_type="table",
                content=content_str,
                table_data=clean_table,
            )

            self.chunks.append(chunk)

    # ------------------------------------------------------------------ #
    # Heading Detection
    # ------------------------------------------------------------------ #

    def _extract_headings_from_text(
        self,
        text: str,
    ):

        """
        Updates current heading tracker
        from normal page text only.
        """

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            if (
                self.HEADING_PATTERN.match(line)
                and self._is_valid_heading(line)
            ):

                self._current_heading = line

    def _is_valid_heading(
        self,
        text: str,
    ) -> bool:

        """
        Prevent OCR/table rows from becoming
        section headings.
        """

        text = text.strip()

        if not text:
            return False

        # avoid giant OCR junk
        if len(text) > 120:
            return False

        # must start with numbering
        if not re.match(
            r"^\d+(\.\d+)*\s+",
            text,
        ):
            return False

        text_lower = text.lower()

        # pharma-specific keywords
        keywords = [
            "review",
            "temperature",
            "humidity",
            "pressure",
            "water",
            "system",
            "environment",
            "batch",
            "formula",
            "quality",
        ]

        if not any(
            k in text_lower
            for k in keywords
        ):
            return False

        # reject OCR-heavy rows
        reject_terms = [
            "usp",
            "ip",
            "mg",
            "gm",
            "ml",
            "cfu",
            "ppb",
            "min",
            "max",
        ]

        if sum(
            term in text_lower
            for term in reject_terms
        ) >= 3:
            return False

        return True

    # ------------------------------------------------------------------ #
    # Table Cleaning
    # ------------------------------------------------------------------ #

    def _clean_table(
        self,
        table: list[list],
    ) -> list[list]:

        """
        Normalizes table rows.
        Removes blank rows.
        """

        cleaned = []

        for row in table:

            if row is None:
                continue

            normalized = []

            for cell in row:

                if cell is None:
                    normalized.append("")
                    continue

                text = str(cell)

                # normalize whitespace
                text = re.sub(
                    r"\s+",
                    " ",
                    text,
                ).strip()

                normalized.append(text)

            # skip fully blank rows
            if not any(normalized):
                continue

            cleaned.append(normalized)

        return cleaned

    # ------------------------------------------------------------------ #
    # Table → Text Conversion
    # ------------------------------------------------------------------ #

    def _table_to_text(
        self,
        table: list[list],
    ) -> str:

        """
        Converts table into readable text.
        """

        if not table:
            return ""

        rows = []

        for row in table:

            rows.append(
                " | ".join(row)
            )

        return "\n".join(rows)
