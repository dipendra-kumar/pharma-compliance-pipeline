"""
Compliance Rules Knowledge Base — Agent 2's source of truth.

Structure:
    COMPLIANCE_RULES[section_key][parameter_key] = {
        "range_text": human-readable rule string,
        "min": lower bound (None if no lower limit),
        "max": upper bound (None if no upper limit),
        "unit": unit of measurement,
    }

Section keys are normalized lowercase strings matched against
extracted section headings during validation.
"""

COMPLIANCE_RULES = {

    # ------------------------------------------------------------------ #
    # Section 6.1 — Temperature, Relative Humidity, Differential Pressure #
    # ------------------------------------------------------------------ #
    "temperature": {
        "temperature_min": {
            "range_text": "No lower limit defined",
            "min": None,
            "max": None,
            "unit": "°C",
        },
        "temperature_max": {
            "range_text": "NMT 25°C",
            "min": None,
            "max": 25.0,
            "unit": "°C",
        },
    },

    "relative_humidity": {
        "relative_humidity_min": {
            "range_text": "No lower limit defined",
            "min": None,
            "max": None,
            "unit": "% RH",
        },
        "relative_humidity_max": {
            "range_text": "NMT 60% RH",
            "min": None,
            "max": 60.0,
            "unit": "% RH",
        },
    },

    "differential_pressure": {
        "differential_pressure_min": {
            "range_text": "NLT 1.5 mm of Wc",
            "min": 1.5,
            "max": None,
            "unit": "mm Wc",
        },
        "differential_pressure_max": {
            "range_text": "No upper limit defined",
            "min": None,
            "max": None,
            "unit": "mm Wc",
        },
    },

    # ------------------------------------------------------------------ #
    # Section 7.0 — Purified Water System                                 #
    # ------------------------------------------------------------------ #
    "microbial_count": {
        "microbial_count_min": {
            "range_text": "No lower limit defined",
            "min": None,
            "max": None,
            "unit": "cfu/ml",
        },
        "microbial_count_max": {
            "range_text": "Alert: NMT 25 cfu/ml | Action: NMT 40 cfu/ml | Standard: NMT 100 cfu/ml",
            "min": None,
            "max": 100.0,   # Standard limit used for PASS/FAIL
            "alert_limit": 25.0,
            "action_limit": 40.0,
            "unit": "cfu/ml",
        },
    },

    "ph": {
        "ph_min": {
            "range_text": "Lower limit: 5",
            "min": 5.0,
            "max": None,
            "unit": "",
        },
        "ph_max": {
            "range_text": "Upper limit: 7",
            "min": None,
            "max": 7.0,
            "unit": "",
        },
    },

    "conductivity": {
        "conductivity_min": {
            "range_text": "No lower limit defined",
            "min": None,
            "max": None,
            "unit": "µs/cm",
        },
        "conductivity_max": {
            "range_text": "NMT 1.3 µs/cm",
            "min": None,
            "max": 1.3,
            "unit": "µs/cm",
        },
    },

    "total_organic_carbon": {
        "toc_min": {
            "range_text": "No lower limit defined",
            "min": None,
            "max": None,
            "unit": "ppb",
        },
        "toc_max": {
            "range_text": "NMT 500 ppb",
            "min": None,
            "max": 500.0,
            "unit": "ppb",
        },
    },

    # ------------------------------------------------------------------ #
    # General Pharma Rules (fallback for common QC parameters)            #
    # ------------------------------------------------------------------ #
    "assay": {
        "assay_result": {
            "range_text": "98.0% – 102.0%",
            "min": 98.0,
            "max": 102.0,
            "unit": "%",
        },
    },

    "dissolution": {
        "dissolution_result": {
            "range_text": "NLT 80% (Q) at 45 minutes",
            "min": 80.0,
            "max": None,
            "unit": "%",
        },
    },

    "water_content": {
        "water_content_result": {
            "range_text": "NMT 0.5%",
            "min": None,
            "max": 0.5,
            "unit": "%",
        },
    },
}


# ------------------------------------------------------------------ #
# Helper: find the rule for a given parameter string                  #
# ------------------------------------------------------------------ #

def find_rule(section_heading: str, parameter: str) -> dict | None:
    """
    Fuzzy-matches section_heading and parameter against COMPLIANCE_RULES.

    Returns the matching rule dict or None if not found.
    """
    section_lower = section_heading.lower()
    param_lower = parameter.lower().replace(" ", "_")

    # Find matching section
    matched_section = None
    for section_key in COMPLIANCE_RULES:
        if section_key in section_lower or section_lower in section_key:
            matched_section = COMPLIANCE_RULES[section_key]
            break

    if not matched_section:
        return None

    # Find matching parameter within section
    for param_key, rule in matched_section.items():
        if param_key in param_lower or param_lower in param_key:
            return rule

    # Fallback: return first rule in section if only one exists
    if len(matched_section) == 1:
        return list(matched_section.values())[0]

    return None


def validate_value(value: float, rule: dict) -> tuple[str, str]:
    """
    Validates a numeric value against a rule dict.

    Returns:
        (status, note) where status is 'PASS' or 'FAIL'
    """
    min_val = rule.get("min")
    max_val = rule.get("max")

    if min_val is not None and value < min_val:
        return "FAIL", f"Value {value} is below minimum {min_val}"

    if max_val is not None and value > max_val:
        return "FAIL", f"Value {value} exceeds maximum {max_val}"

    # Alert/Action limit checks (warning only, still PASS on standard limit)
    note = "Within compliance range"
    alert = rule.get("alert_limit")
    action = rule.get("action_limit")

    if alert and value > alert:
        note = f"Exceeds alert limit ({alert}) — investigation recommended"
    if action and value > action:
        note = f"Exceeds action limit ({action}) — corrective action required"

    return "PASS", note
