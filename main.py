"""
Pharma Compliance Pipeline - End-to-end runner.

Stages:
    1. DocumentProcessor   - PDF ingestion + table extraction
    2. ExtractionAgent     - structured numerical extraction
    3. ValidationAgent     - compliance validation
    4. AnalyticalAgent     - statistics, trends, insights, charts

Outputs:
    output/agent1_output.json
    output/agent2_output.json
    output/agent3_output.json
    output/charts/*.png
    output/pipeline_output.json   (combined PipelineOutput)
"""

import json
import os
import sys

from agents.analytical_agent import AnalyticalAgent
from agents.extraction_agent import ExtractionAgent
from agents.validation_agent import ValidationAgent
from models.schemas import PipelineOutput
from pipeline.document_processor import DocumentProcessor
from pipeline.vector_store import VectorStore


PDF_PATH = "sample_extract_call.pdf"
OUTPUT_DIR = "output"


def main(pdf_path: str = PDF_PATH) -> PipelineOutput:

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------ #
    # Stage 1 - Document processing + indexing
    # ------------------------------------------------------------ #

    print("=" * 60)
    print("STAGE 1: Document Processing")
    print("=" * 60)

    processor = DocumentProcessor(pdf_path)
    processor.process()

    table_chunks = [
        c for c in processor.get_table_chunks()
        if len(c.table_data) >= 2
    ]

    print(f"Table chunks to process: {len(table_chunks)}")

    store = VectorStore()
    store.reset()
    store.index(processor.chunks if hasattr(processor, "chunks") else [])

    # ------------------------------------------------------------ #
    # Stage 2 - Agent 1: Extraction
    # ------------------------------------------------------------ #

    print("\n" + "=" * 60)
    print("STAGE 2: Agent 1 - Extraction")
    print("=" * 60)

    extraction_agent = ExtractionAgent()
    extracted_records = extraction_agent.run(table_chunks)

    with open(f"{OUTPUT_DIR}/agent1_output.json", "w") as f:
        json.dump([r.model_dump() for r in extracted_records], f, indent=2)

    # ------------------------------------------------------------ #
    # Stage 3 - Agent 2: Validation
    # ------------------------------------------------------------ #

    print("\n" + "=" * 60)
    print("STAGE 3: Agent 2 - Validation")
    print("=" * 60)

    validation_agent = ValidationAgent(vector_store=store, use_rag=False)
    validation_records = validation_agent.run(extracted_records)

    with open(f"{OUTPUT_DIR}/agent2_output.json", "w") as f:
        json.dump([r.model_dump() for r in validation_records], f, indent=2)

    # ------------------------------------------------------------ #
    # Stage 4 - Agent 3: Analytics
    # ------------------------------------------------------------ #

    print("\n" + "=" * 60)
    print("STAGE 4: Agent 3 - Analytics")
    print("=" * 60)

    analytical_agent = AnalyticalAgent(output_dir=OUTPUT_DIR)
    analytical_summaries = analytical_agent.run(validation_records)

    with open(f"{OUTPUT_DIR}/agent3_output.json", "w") as f:
        json.dump([s.model_dump() for s in analytical_summaries], f, indent=2)

    # ------------------------------------------------------------ #
    # Combined PipelineOutput
    # ------------------------------------------------------------ #

    total_pass = sum(1 for r in validation_records if r.validation_status == "PASS")
    total_fail = sum(1 for r in validation_records if r.validation_status == "FAIL")

    pipeline_output = PipelineOutput(
        document_name=os.path.basename(pdf_path),
        total_records_extracted=len(extracted_records),
        total_pass=total_pass,
        total_fail=total_fail,
        validation_report=validation_records,
        analytical_summaries=analytical_summaries,
    )

    with open(f"{OUTPUT_DIR}/pipeline_output.json", "w") as f:
        json.dump(pipeline_output.model_dump(), f, indent=2)

    # ------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------ #

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Document            : {pipeline_output.document_name}")
    print(f"  Extracted records   : {pipeline_output.total_records_extracted}")
    print(f"  PASS                : {pipeline_output.total_pass}")
    print(f"  FAIL                : {pipeline_output.total_fail}")
    print(f"  Analytical summaries: {len(analytical_summaries)}")
    print(f"\nOutputs written to {OUTPUT_DIR}/")

    return pipeline_output


if __name__ == "__main__":

    path = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    main(path)
