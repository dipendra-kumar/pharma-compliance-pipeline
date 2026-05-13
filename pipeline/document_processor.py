"""
Document Processor — ingests a PDF and extracts both raw text
and structured tables using pdfplumber.

Output is a list of DocumentChunk objects, each representing
either a text block or a table from a specific page/section.
"""

import pdfplumber
import re
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """
    Represents a single chunk of content extracted from the PDF.
    Could be a text block or a table.
    """
    chunk_id: str
    page_number: int
    section_heading: str
    chunk_type: str                      # "text" or "table"
    content: str                         # raw text representation
    table_data: list[list] = field(default_factory=list)   # raw rows if table


class DocumentProcessor:
    """
    Loads a PDF and extracts text + tables page by page.
    Detects section headings to give each chunk its context.
    """

    # Regex to detect headings like "3.1 List of batches" or "6.0 Review of..."
    HEADING_PATTERN = re.compile(r"^\d+(\.\d+)?\s+[A-Z][^\n]{5,}")

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.chunks: list[DocumentChunk] = []
        self._current_heading = "General"

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def process(self) -> list[DocumentChunk]:
        """
        Main entry point. Opens the PDF and processes every page.
        Returns a flat list of DocumentChunks.
        """
        self.chunks = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                self._process_page(page, page_num)

        print(f"[DocumentProcessor] Extracted {len(self.chunks)} chunks "
              f"from {self.pdf_path}")
        return self.chunks

    def get_text_chunks(self) -> list[DocumentChunk]:
        return [c for c in self.chunks if c.chunk_type == "text"]

    def get_table_chunks(self) -> list[DocumentChunk]:
        return [c for c in self.chunks if c.chunk_type == "table"]

    def get_full_text(self) -> str:
        """Returns entire document as one string — used by RAG indexer."""
        return "\n\n".join(c.content for c in self.chunks)

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _process_page(self, page, page_num: int):
        """Extracts text blocks and tables from a single page."""

        # --- Extract tables first (pdfplumber is better at isolating them) ---
        tables = page.extract_tables()
        table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

        for idx, table in enumerate(tables):
            if not table:
                continue

            clean_table = self._clean_table(table)
            if not clean_table:
                continue

            content_str = self._table_to_text(clean_table)
            self._update_heading(content_str)

            chunk = DocumentChunk(
                chunk_id=f"p{page_num}_table{idx}",
                page_number=page_num,
                section_heading=self._current_heading,
                chunk_type="table",
                content=content_str,
                table_data=clean_table,
            )
            self.chunks.append(chunk)

        # --- Extract remaining text (outside table bounding boxes) ---
        raw_text = page.extract_text()
        if raw_text:
            # Update heading tracker from page text
            for line in raw_text.splitlines():
                if self.HEADING_PATTERN.match(line.strip()):
                    self._current_heading = line.strip()

            text_chunk = DocumentChunk(
                chunk_id=f"p{page_num}_text",
                page_number=page_num,
                section_heading=self._current_heading,
                chunk_type="text",
                content=raw_text.strip(),
            )
            self.chunks.append(text_chunk)

    def _update_heading(self, text: str):
        """Checks text for a section heading and updates tracker."""
        for line in text.splitlines():
            if self.HEADING_PATTERN.match(line.strip()):
                self._current_heading = line.strip()
                break

    def _clean_table(self, table: list[list]) -> list[list]:
        """
        Removes fully empty rows and normalizes cell values to strings.
        """
        cleaned = []
        for row in table:
            if row is None:
                continue
            normalized = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(cell != "" for cell in normalized):  # skip blank rows
                cleaned.append(normalized)
        return cleaned

    def _table_to_text(self, table: list[list]) -> str:
        """
        Converts a 2D table into a readable pipe-delimited string.
        Example:
            Month | Area | Temp Min | Temp Max
            Mar-23 | Blender III | 19.5 | 23.1
        """
        if not table:
            return ""
        rows = [" | ".join(row) for row in table]
        return "\n".join(rows)
