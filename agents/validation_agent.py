"""
Agent 2 — Extraction Quality + Compliance Validation Agent

Responsibilities:
- Validates extraction quality from Agent 1
- Detects malformed extraction outputs
- Detects OCR leakage / bad parsing
- Validates compliance against pharma rules
- Returns ValidationRecord objects
"""

from models.schemas import (
    ExtractedRecord,
    ValidationRecord,
)

from rules.compliance_rules import (
    find_rule,
    validate_value,
)

# ------------------------------------------------------------------ #
# Parameter normalization map
# ------------------------------------------------------------------ #

PARAM_TO_SECTION = {

    # Temperature
    "temperature min": "temperature",
    "temperature max": "temperature",

    # Relative Humidity
    "relative humidity min": "relative_humidity",
    "relative humidity max": "relative_humidity",
    "rh min": "relative_humidity",
    "rh max": "relative_humidity",

    # Differential Pressure
    "differential pressure min": "differential_pressure",
    "differential pressure max": "differential_pressure",

    # Microbial Count
    "microbial count min": "microbial_count",
    "microbial count max": "microbial_count",

    # pH
    "ph min": "ph",
    "ph max": "ph",

    # Conductivity
    "conductivity min": "conductivity",
    "conductivity max": "conductivity",

    # Total Organic Carbon
    "total organic carbon min": "total_organic_carbon",
    "total organic carbon max": "total_organic_carbon",
    "toc min": "total_organic_carbon",
    "toc max": "total_organic_carbon",

    # Assay
    "assay": "assay",

    # Dissolution
    "dissolution": "dissolution",
}


class ValidationAgent:

    def __init__(self, vector_store=None, use_rag=False):

        self.name = "ValidationAgent"

        # kept for compatibility with your existing test files
        self.vector_store = vector_store
        self.use_rag = use_rag

    # ------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------ #

    def run(
        self,
        records: list[ExtractedRecord],
    ) -> list[ValidationRecord]:

        validated = []

        for record in records:

            # ------------------------------------------------ #
            # Step 1 — Extraction Quality Validation
            # ------------------------------------------------ #

            quality_issue = self._check_quality(record)

            if quality_issue:

                validated.append(
                    ValidationRecord(
                        section_heading=record.section_heading,
                        table_name=record.table_name,
                        parameter=record.parameter,
                        extracted_value=record.extracted_value,
                        unit=record.unit,
                        row_context=record.row_context,
                        compliance_range="N/A",
                        validation_status="QUALITY_ISSUE",
                        validation_note=quality_issue,
                    )
                )

                continue

            # ------------------------------------------------ #
            # Step 2 — Compliance Validation
            # ------------------------------------------------ #

            result = self._validate_record(record)

            validated.append(
                ValidationRecord(
                    section_heading=record.section_heading,
                    table_name=record.table_name,
                    parameter=record.parameter,
                    extracted_value=record.extracted_value,
                    unit=record.unit,
                    row_context=record.row_context,
                    compliance_range=result["compliance_range"],
                    validation_status=result["validation_status"],
                    validation_note=result["note"],
                )
            )

        # ------------------------------------------------------------ #
        # Summary
        # ------------------------------------------------------------ #

        pass_count = sum(
            1 for r in validated
            if r.validation_status == "PASS"
        )

        fail_count = sum(
            1 for r in validated
            if r.validation_status == "FAIL"
        )

        quality_count = sum(
            1 for r in validated
            if r.validation_status == "QUALITY_ISSUE"
        )

        skipped_count = sum(
            1 for r in validated
            if r.validation_status == "SKIPPED"
        )

        print(f"\n[{self.name}] Validation complete:")
        print(f"  PASS            : {pass_count}")
        print(f"  FAIL            : {fail_count}")
        print(f"  QUALITY ISSUES  : {quality_count}")
        print(f"  SKIPPED         : {skipped_count}")

        return validated

    # ------------------------------------------------------------ #
    # Extraction Quality Checks
    # ------------------------------------------------------------ #

    def _check_quality(
        self,
        record: ExtractedRecord,
    ) -> str | None:

        param = record.parameter.lower().strip()

        # Fake placeholder columns
        if "column_" in param:
            return (
                "Extractor generated placeholder column name. "
                "Header mapping likely failed."
            )

        # Month/date parsed as numeric
        if (
            param == "month"
            and isinstance(record.extracted_value, (int, float))
        ):
            return (
                "Month/date value incorrectly extracted "
                "as numeric measurement."
            )

        # Likely OCR ID extraction
        if (
            isinstance(record.extracted_value, (int, float))
            and record.extracted_value > 100000
        ):
            return (
                "Value appears to be an item code or OCR ID "
                "instead of a compliance measurement."
            )

        # Corrupted section heading
        if len(record.section_heading) > 120:
            return (
                "Section heading appears corrupted "
                "or merged with table content."
            )

        # Missing units
        if (
            "temperature" in param
            and not record.unit
        ):
            return (
                "Temperature parameter missing °C unit."
            )

        # Impossible humidity values
        if (
            "humidity" in param
            and isinstance(record.extracted_value, (int, float))
            and record.extracted_value > 100
        ):
            return (
                "Relative humidity cannot exceed 100%."
            )

        # Impossible pH values
        if (
            "ph" in param
            and isinstance(record.extracted_value, (int, float))
            and (
                record.extracted_value < 0
                or record.extracted_value > 14
            )
        ):
            return (
                "pH value outside valid scientific range."
            )

        return None

    # ------------------------------------------------------------ #
    # Compliance Validation
    # ------------------------------------------------------------ #

    def _validate_record(
        self,
        record: ExtractedRecord,
    ) -> dict:

        param_lower = (
            record.parameter
            .lower()
            .strip()
        )

        # Map parameter to section
        section_key = None

        # normalize parameter
        param_lower = (
            record.parameter
            .lower()
            .strip()
        )
        # ------------------------------------------------ #
        # Normalize common aliases
        # ------------------------------------------------ #
        
        param_lower = (
            param_lower
            .replace("rh", "relative_humidity")
            .replace("temp", "temperature")
            .replace("dp", "differential_pressure")
            .replace("toc", "total_organic_carbon")
        )
        
        param_lower = param_lower.replace(" ", "_")
       
       # direct keyword mapping
        if "temperature" in param_lower:
            section_key = "temperature"

        elif (
            "relative humidity" in param_lower
            or "rh" in param_lower
        ):
            section_key = "relative_humidity"

        elif "differential pressure" in param_lower:
            section_key = "differential_pressure"

        elif "ph" in param_lower:
            section_key = "ph"

        elif "conductivity" in param_lower:
            section_key = "conductivity"

        elif (
            "microbial" in param_lower
            or "cfu" in param_lower
        ):
            section_key = "microbial_count"

        elif (
            "toc" in param_lower
            or "total organic carbon" in param_lower
        ):
            section_key = "total_organic_carbon"

        elif "assay" in param_lower:
            section_key = "assay"

        elif "dissolution" in param_lower:
            section_key = "dissolution"        # No rule found
        if not section_key:
            return {
                "compliance_range": "No rule found",
                "validation_status": "SKIPPED",
                "note": (
                    f"No compliance rule mapped "
                    f"for parameter: {record.parameter}"
                ),
            }

        # Find rule
        rule = find_rule(
            section_key,
            param_lower,
        )

        if not rule:
            return {
                "compliance_range": "No rule found",
                "validation_status": "SKIPPED",
                "note": (
                    f"Rule lookup failed "
                    f"for parameter: {record.parameter}"
                ),
            }

        # Validate numeric value
        try:

            value = float(record.extracted_value)

            status, note = validate_value(
                value,
                rule,
            )

            return {
                "compliance_range": rule[
                    "range_text"
                ],
                "validation_status": status,
                "note": note,
            }

        except Exception:

            return {
                "compliance_range": rule.get(
                    "range_text",
                    "Unknown",
                ),
                "validation_status": "SKIPPED",
                "note": (
                    f"Could not validate value: "
                    f"{record.extracted_value}"
                ),
            }

