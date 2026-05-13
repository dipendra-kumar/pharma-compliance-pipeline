
"""
Agent 1 — Extraction Agent

Responsibilities:
- Receives table chunks from the document processor
- Uses Claude to extract structured numerical values from each table
- Returns a list of ExtractedRecord objects
"""

import json
from models.schemas import ExtractedRecord
from utils.llm_client import call_claude
from pipeline.document_processor import DocumentChunk


# ------------------------------------------------------------------ #
# System prompt — defines the extractor agent's role and extraction rules         #
# ------------------------------------------------------------------ #

EXTRACTION_SYSTEM_PROMPT = """
You are a pharmaceutical QC data extraction specialist.

Your job is to extract structured numerical data from compliance tables
found in Product Quality Review (PQR) documents.

RULES:
1. Extract EVERY numerical value from the table — do not skip any rows.
2. For tables with Min/Max columns, create SEPARATE records for Min and Max values.
3. Always capture the row context (e.g., Month + Area, or Month alone).
4. Identify the correct parameter name from column headers.
5. Capture units from column headers (e.g., °C, % RH, cfu/ml, ppb).
6. If a cell is empty or "N/A", skip it.

OUTPUT FORMAT:
Return a JSON array of objects. Each object must have exactly these fields:
{
  "section_heading": "the section title this table belongs to",
  "table_name": "a short identifier for this table",
  "parameter": "exact parameter name with Min or Max suffix if applicable",
  "extracted_value": <number>,
  "unit": "unit string or empty string",
  "row_context": "e.g. Mar-23 | Blender III or Feb-23"
}

Return ONLY the JSON array. No explanation, no markdown fences.
"""


class ExtractionAgent:
    """
    Agent 1: Extracts numerical compliance data from table chunks.
    """

    def __init__(self):
        self.name = "ExtractionAgent"

    def run(self, table_chunks: list[DocumentChunk]) -> list[ExtractedRecord]:
        """
        Processes all table chunks and returns a flat list of ExtractedRecords.

        Args:
            table_chunks: Table chunks from DocumentProcessor

        Returns:
            List of ExtractedRecord objects
        """
        all_records: list[ExtractedRecord] = []

        for chunk in table_chunks:
            print(f"[{self.name}] Processing: {chunk.chunk_id} | "
                  f"Section: {chunk.section_heading[:60]}")

            records = self._extract_from_chunk(chunk)
            all_records.extend(records)
            print(f"[{self.name}] → Extracted {len(records)} records")

        print(f"\n[{self.name}] Total extracted: {len(all_records)} records")
        return all_records

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _extract_from_chunk(self, chunk: DocumentChunk) -> list[ExtractedRecord]:
        """
        Sends a single table chunk to Claude and parses the response
        into ExtractedRecord objects.
        """
        user_message = f"""
Section Heading: {chunk.section_heading}
Table ID: {chunk.chunk_id}
Page: {chunk.page_number}

Table Content:
{chunk.content}

Extract all numerical compliance values from this table following the rules.
"""
        try:
            raw = call_claude(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_message=user_message,
                expect_json=True,
            )

            # raw is a list of dicts
            records = []
            for item in raw:
                try:
                    # Coerce extracted_value to float
                    item["extracted_value"] = float(str(item["extracted_value"])
                                                    .replace(",", "").strip())
                    records.append(ExtractedRecord(**item))
                except (ValueError, TypeError, KeyError) as e:
                    print(f"[{self.name}] Skipping malformed record: {item} | {e}")

            return records

        except json.JSONDecodeError as e:
            print(f"[{self.name}] JSON parse error on {chunk.chunk_id}: {e}")
            return []
        except Exception as e:
            print(f"[{self.name}] Error on {chunk.chunk_id}: {e}")
            return []
