"""
Agent 1 — Deterministic Extraction Agent

Responsibilities:
- Receives table chunks from DocumentProcessor
- Parses structured numerical values directly
- Returns ExtractedRecord objects
"""

import re

from models.schemas import ExtractedRecord
from pipeline.document_processor import DocumentChunk


class ExtractionAgent:

    def __init__(self):
        self.name = "ExtractionAgent"

    # ------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------ #

    def run(
        self,
        table_chunks: list[DocumentChunk],
    ) -> list[ExtractedRecord]:

        all_records = []

        for chunk in table_chunks:

            print(
                f"[{self.name}] "
                f"Processing {chunk.chunk_id}"
            )

            try:

                records = self._extract_from_table(chunk)

                all_records.extend(records)

                print(
                    f"[{self.name}] "
                    f"→ Extracted {len(records)} records"
                )

            except Exception as e:

                print(
                    f"[{self.name}] "
                    f"Error processing {chunk.chunk_id}: {e}"
                )

        print(
            f"\n[{self.name}] "
            f"Total extracted: {len(all_records)} records"
        )

        return all_records

    # ------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------ #

    def _extract_from_table(
        self,
        chunk: DocumentChunk,
    ) -> list[ExtractedRecord]:

        records = []

        table = chunk.table_data

        if len(table) < 4:
            return []

        # ------------------------------------------------------------ #
        # Detect header rows
        # ------------------------------------------------------------ #

        header_rows = table[:4]

        max_cols = max(
            len(r)
            for r in header_rows
        )

        # normalize rows
        normalized_headers = []

        for row in header_rows:

            normalized_row = [
                str(x).strip()
                for x in row
            ]

            while len(normalized_row) < max_cols:
                normalized_row.append("")

            normalized_headers.append(
                normalized_row
            )

        # ------------------------------------------------------------ #
        # Build hierarchical headers
        # ------------------------------------------------------------ #

        merged_headers = []

        hierarchy_memory = [""] * 4

        for col_idx in range(max_cols):

            parts = []

            for row_idx in range(4):

                cell = normalized_headers[row_idx][col_idx]

                if cell:
                    hierarchy_memory[row_idx] = cell

                current = hierarchy_memory[row_idx]

                # skip noisy labels
                if current.lower() in [
                    "observed values",
                    "limit",
                    "limits",
                    "",
                ]:
                    continue

                parts.append(current)

            # clean hierarchy
            cleaned_parts = []

            for p in parts:

                p = (
                    p.lower()
                    .replace("\n", " ")
                    .replace(".", "")
                    .strip()
                )

                if p not in cleaned_parts:
                    cleaned_parts.append(p)

            final_header = "_".join(cleaned_parts)

            final_header = (
                final_header
                .replace(" ", "_")
                .replace("__", "_")
            )

            merged_headers.append(
                final_header
            )

        # ------------------------------------------------------------ #
        # Process data rows
        # ------------------------------------------------------------ #

        data_rows = table[4:]

        for row in data_rows:

            row_context = self._build_row_context(row)

            for col_idx, cell in enumerate(row):

                if col_idx >= len(merged_headers):
                    continue

                parameter = merged_headers[col_idx]

                if not parameter:
                    continue

                # skip metadata columns
                if self._should_skip_parameter(
                    parameter
                ):
                    continue

                numeric_value = self._parse_numeric(
                    cell
                )

                if numeric_value is None:
                    continue

                # skip OCR junk
                if self._looks_like_code(cell):
                    continue

                # canonicalize parameter names
                parameter = self._canonicalize_parameter(
                    parameter
                )

                unit = self._extract_unit(
                    parameter
                )

                records.append(
                    ExtractedRecord(
                        section_heading=chunk.section_heading,
                        table_name=chunk.chunk_id,
                        parameter=parameter,
                        extracted_value=numeric_value,
                        unit=unit,
                        row_context=row_context,
                    )
                )

        return records

    def _parse_numeric(
        self,
        value: str,
    ) -> float | None:

        if value is None:
            return None

        value = str(value).strip()

        invalid_values = [
            "",
            "N/A",
            "Nil",
            "Q.S.",
            "-",
        ]

        if value in invalid_values:
            return None

        # skip dates
        if re.match(
            r"^[A-Za-z]{3}-\d{2}$",
            value,
        ):
            return None

        # skip item codes
        if self._looks_like_code(value):
            return None

        value = (
            value
            .replace(",", "")
            .replace("%", "")
        )

        # strict numeric match
        match = re.fullmatch(
            r"-?\d+(\.\d+)?",
            value,
        )

        if not match:
            return None

        try:
            return float(value)
        except:
            return None

    def _build_row_context(
        self,
        row: list,
    ) -> str:

        context_parts = []

        for cell in row[:2]:

            if cell and str(cell).strip():
                context_parts.append(
                    str(cell).strip()
                )

        return " | ".join(context_parts)

    def _extract_unit(
        self,
        parameter: str,
    ) -> str:

        parameter = parameter.lower()

        if "°c" in parameter:
            return "°C"

        if "% rh" in parameter:
            return "% RH"

        if "cfu" in parameter:
            return "cfu/ml"

        if "ppb" in parameter:
            return "ppb"

        if "ph" in parameter:
            return "pH"

        return ""

    def _canonicalize_parameter(
        self,
        parameter: str,
    ) -> str:

        p = parameter.lower()

        # microbial count
        if "microbial" in p:

            if "min" in p:
                return "microbial_count_min"

            if "max" in p:
                return "microbial_count_max"

        # pH
        if "ph" in p:

            if "min" in p:
                return "ph_min"

            if "max" in p:
                return "ph_max"

        # conductivity
        if "conductivity" in p:

            if "min" in p:
                return "conductivity_min"

            if "max" in p:
                return "conductivity_max"

        # TOC
        if (
            "organic" in p
            or "toc" in p
        ):

            if "min" in p:
                return "toc_min"

            if "max" in p:
                return "toc_max"

        # temperature
        if "temperature" in p:

            if "min" in p:
                return "temperature_min"

            if "max" in p:
                return "temperature_max"

        # RH
        if (
            "humidity" in p
            or "rh" in p
        ):

            if "min" in p:
                return "relative_humidity_min"

            if "max" in p:
                return "relative_humidity_max"

        # differential pressure
        if (
            "differential" in p
            or "pressure" in p
        ):

            if "min" in p:
                return "differential_pressure_min"

            if "max" in p:
                return "differential_pressure_max"

        return p

    def _should_skip_parameter(
        self,
        parameter: str,
    ) -> bool:

        parameter = parameter.lower()

        skip_terms = [
            "month",
            "batch",
            "date",
            "reference",
            "document",
            "sr no",
            "ingredient",
            "item code",
            "grade",
            "pack",
            "mfr",
            "bmr",
            "mpr",
            "bpr",
            "remark",
            "result",
            "sign",
            "verify"
        ]

        return any(
            term in parameter
            for term in skip_terms
        )

    def _looks_like_code(
        self,
        value: str,
    ) -> bool:

        value = str(value).strip()

        # alphanumeric item codes
        if (
            re.search(r"[A-Za-z]", value)
            and re.search(r"\d", value)
        ):
            return True

        # long IDs
        if len(value) > 12:
            return True

        return False
