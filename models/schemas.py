from pydantic import BaseModel, Field
from typing import Optional


class ExtractedRecord(BaseModel):
    """
    Output of Agent 1 (Extraction Agent).
    Represents a single numerical value pulled from a table.
    """
    section_heading: str = Field(..., description="Section heading where the data was found")
    table_name: str = Field(..., description="Table identifier or name")
    parameter: str = Field(..., description="The QC parameter name e.g. Temperature Max")
    extracted_value: float | str = Field(..., description="The numerical or string value extracted")
    unit: Optional[str] = Field(None, description="Unit of measurement e.g. °C, % RH")
    row_context: Optional[str] = Field(None, description="Extra context like month/area for this row")

class ValidationRecord(BaseModel):

    section_heading: str
    table_name: str
    parameter: str
    extracted_value: float | str
    unit: Optional[str]
    row_context: Optional[str]

    compliance_range: str

    validation_status: str
    # PASS / FAIL / QUALITY_ISSUE / SKIPPED

    validation_note: Optional[str] = None

class AnalyticalSummary(BaseModel):
    """
    Output of Agent 3 (Analytical Agent).
    Statistical summary for a group of validated values.
    """
    parameter: str
    section: str
    unit: Optional[str]
    count: int
    mean: float
    median: float
    mode: Optional[float] = Field(None, description="Mode of observations; None if no unique mode")
    std_dev: float
    min_value: float
    max_value: float
    cpk: Optional[float] = Field(None, description="Process Capability Index if limits are numeric")
    trend: str = Field("INSUFFICIENT_DATA", description="STABLE / INCREASING / DECREASING / INSUFFICIENT_DATA")
    pass_count: int
    fail_count: int
    insight: str = Field(..., description="Human-readable pharmacological/analytical insight")
    chart_files: list[str] = Field(default_factory=list, description="Paths to generated PNG charts for this parameter")


class PipelineOutput(BaseModel):
    """
    Final combined output of the entire pipeline.
    """
    document_name: str
    total_records_extracted: int
    total_pass: int
    total_fail: int
    validation_report: list[ValidationRecord]
    analytical_summaries: list[AnalyticalSummary]
