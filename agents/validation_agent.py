"""
Agent 2 — Validation Agent

Responsibilities:
- Receives ExtractedRecords from Agent 1
- Validates each value against compliance_rules.py (primary check)
- Re-validates ambiguous/failed cases against raw document via RAG (secondary check)
- Returns a list of ValidationRecord objects with PASS/FAIL status
"""

from models.schemas import ExtractedRecord, ValidationRecord
from rules.compliance_rules import find_rule, validate_value
from pipeline.vector_store import VectorStore
from utils.llm_client import call_claude
import time

# Delay between RAG LLM calls
RAG_CALL_DELAY = 5  # seconds

# ------------------------------------------------------------------ #
# Parameter normalisation map                                         #
# Maps extracted parameter names → compliance rule section keys      #
# ------------------------------------------------------------------ #

PARAM_TO_SECTION = {
    # Temperature
    "temperature min":       "temperature",
    "temperature max":       "temperature",
    # Relative Humidity
    "relative humidity min": "relative_humidity",
    "relative humidity max": "relative_humidity",
    "rh min":                "relative_humidity",
    "rh max":                "relative_humidity",
    # Differential Pressure
    "differential pressure min": "differential_pressure",
    "differential pressure max": "differential_pressure",
    # Microbial Count
    "microbial count min.":  "microbial_count",
    "microbial count max.":  "microbial_count",
    "microbial count min":   "microbial_count",
    "microbial count max":   "microbial_count",
    # pH
    "ph min.":               "ph",
    "ph max.":               "ph",
    "ph min":                "ph",
    "ph max":                "ph",
    # Conductivity
    "conductivity min.":     "conductivity",
    "conductivity max.":     "conductivity",
    "conductivity min":      "conductivity",
    "conductivity max":      "conductivity",
    # Total Organic Carbon
    "total organic carbon min.": "total_organic_carbon",
    "total organic carbon max.": "total_organic_carbon",
    "total organic carbon min":  "total_organic_carbon",
    "total organic carbon max":  "total_organic_carbon",
    "toc min":               "total_organic_carbon",
    "toc max":               "total_organic_carbon",
    # Assay
    "assay result":          "assay",
    "assay":                 "assay",
    # Dissolution
    "dissolution result":    "dissolution",
    "dissolution":           "dissolution",
}

# Which parameters use the "min" rule vs "max" rule
MIN_PARAMS = {
    "temperature min", "relative humidity min", "rh min",
    "differential pressure min",
    "microbial count min", "microbial count min.",
    "ph min", "ph min.",
    "conductivity min", "conductivity min.",
    "total organic carbon min", "total organic carbon min.",
    "toc min",
}

MAX_PARAMS = {
    "temperature max", "relative humidity max", "rh max",
    "differential pressure max",
    "microbial count max", "microbial count max.",
    "ph max", "ph max.",
    "conductivity max", "conductivity max.",
    "total organic carbon max", "total organic carbon max.",
    "toc max",
}


# ------------------------------------------------------------------ #
# RAG re-validation prompt                                            #
# ------------------------------------------------------------------ #

RAG_SYSTEM_PROMPT = """
You are a pharmaceutical compliance auditor performing a secondary verification.

You will be given:
1. An extracted data record (parameter name, value, context)
2. Relevant raw document excerpts retrieved from the source document

Your task: Confirm whether the extracted value is correctly read from the document.

Respond ONLY with a JSON object:
{
  "value_confirmed": true or false,
  "correct_value": <number or null if unconfirmed>,
  "note": "brief explanation"
}

No markdown, no extra text.
"""


class ValidationAgent:
    """
    Agent 2: Validates extracted records against compliance rules and
    optionally re-validates against the raw document via RAG.
    """

    def __init__(self, vector_store: VectorStore, use_rag: bool = True):
        self.name = "ValidationAgent"
        self.vector_store = vector_store
        self.use_rag = use_rag

    def run(self, records: list[ExtractedRecord]) -> list[ValidationRecord]:
        """
        Validates all records. Returns list of ValidationRecord objects.
        """
        validated: list[ValidationRecord] = []
        rag_call_count = 0

        for record in records:
            result = self._validate_record(record)

            # RAG re-check: only on FAIL or when rule not found
            if self.use_rag and (
                result["validation_status"] == "FAIL"
                or result["compliance_range"] == "No rule found"
            ):
                result = self._rag_revalidate(record, result)
                rag_call_count += 1
                time.sleep(RAG_CALL_DELAY)

            validated.append(ValidationRecord(
                section_heading=record.section_heading,
                table_name=record.table_name,
                parameter=record.parameter,
                extracted_value=record.extracted_value,
                unit=record.unit,
                row_context=record.row_context,
                compliance_range=result["compliance_range"],
                validation_status=result["validation_status"],
                validation_note=result["note"],
            ))

        pass_count = sum(1 for r in validated if r.validation_status == "PASS")
        fail_count = sum(1 for r in validated if r.validation_status == "FAIL")
        skipped = sum(1 for r in validated if r.validation_status == "SKIPPED")

        print(f"\n[{self.name}] Validation complete:")
        print(f"  PASS    : {pass_count}")
        print(f"  FAIL    : {fail_count}")
        print(f"  SKIPPED : {skipped}")
        print(f"  RAG calls made: {rag_call_count}")

        return validated

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _validate_record(self, record: ExtractedRecord) -> dict:
        """
        Primary validation: looks up rule in compliance_rules.py
        and runs validate_value().
        """
        param_lower = record.parameter.lower().strip()

        # Find section key from parameter name
        section_key = PARAM_TO_SECTION.get(param_lower)

        if not section_key:
            # Try partial match
            for key in PARAM_TO_SECTION:
                if key in param_lower or param_lower in key:
                    section_key = PARAM_TO_SECTION[key]
                    break

        if not section_key:
            return {
                "compliance_range": "No rule found",
                "validation_status": "SKIPPED",
                "note": f"No compliance rule mapped for parameter: {record.parameter}",
            }

        # Determine which sub-rule to use (min vs max side of the range)
        if param_lower in MIN_PARAMS:
            param_key = f"{section_key}_min"
        elif param_lower in MAX_PARAMS:
            param_key = f"{section_key}_max"
        else:
            # Default: use max rule for single-value parameters
            param_key = f"{section_key}_result"

        # Find the rule
        rule = find_rule(section_key, param_key)
        if not rule:
            rule = find_rule(section_key, section_key)

        if not rule:
            return {
                "compliance_range": "No rule found",
                "validation_status": "SKIPPED",
                "note": f"Rule lookup failed for {section_key}/{param_key}",
            }

        # Skip validation if neither limit is defined (e.g. min temperature)
        if rule.get("min") is None and rule.get("max") is None:
            return {
                "compliance_range": rule["range_text"],
                "validation_status": "SKIPPED",
                "note": "No numeric limit defined for this side of the range",
            }

        # Validate
        try:
            value = float(record.extracted_value)
            status, note = validate_value(value, rule)
            return {
                "compliance_range": rule["range_text"],
                "validation_status": status,
                "note": note,
            }
        except (ValueError, TypeError):
            return {
                "compliance_range": rule.get("range_text", "Unknown"),
                "validation_status": "SKIPPED",
                "note": f"Could not convert value to float: {record.extracted_value}",
            }

    def _rag_revalidate(self, record: ExtractedRecord, current_result: dict) -> dict:
        """
        Secondary validation using RAG: retrieves relevant document chunks
        and asks the LLM to confirm the extracted value.
        If confirmed, keeps current result. If not, updates value and re-validates.
        """
        query = (
            f"{record.parameter} {record.row_context} "
            f"{record.section_heading}"
        )
        hits = self.vector_store.query(query, n_results=3)
        context = "\n\n---\n\n".join(h["content"] for h in hits)

        user_message = f"""
Extracted Record:
  Parameter   : {record.parameter}
  Value       : {record.extracted_value}
  Unit        : {record.unit}
  Row Context : {record.row_context}
  Section     : {record.section_heading}

Document Excerpts:
{context}

Is the extracted value correct based on the document excerpts?
"""
        try:
            rag_result = call_claude(
                system_prompt=RAG_SYSTEM_PROMPT,
                user_message=user_message,
                expect_json=True,
                max_tokens=256,
            )

            if not rag_result.get("value_confirmed", True):
                correct_val = rag_result.get("correct_value")
                note = rag_result.get("note", "RAG correction applied")
                if correct_val is not None:
                    # Re-validate with corrected value
                    corrected = ExtractedRecord(
                        section_heading=record.section_heading,
                        table_name=record.table_name,
                        parameter=record.parameter,
                        extracted_value=float(correct_val),
                        unit=record.unit,
                        row_context=record.row_context,
                    )
                    result = self._validate_record(corrected)
                    result["note"] = f"RAG corrected value from {record.extracted_value} to {correct_val}. {note}"
                    return result

            # Value confirmed — keep original result but add RAG note
            current_result["note"] = (
                f"{current_result.get('note', '')} | "
                f"RAG confirmed: {rag_result.get('note', 'value verified')}"
            )
            return current_result

        except Exception as e:
            print(f"[{self.name}] RAG revalidation error: {e}")
            return current_result
