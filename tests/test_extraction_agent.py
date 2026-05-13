"""
Test Agent 1 in isolation.
Usage: python test_agent1.py
"""

import json
import os

from pipeline.document_processor import DocumentProcessor
from agents.extraction_agent import ExtractionAgent

PDF_PATH = "sample_extract_call.pdf"

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# ------------------------------------------------------------ #
# Ingest document
# ------------------------------------------------------------ #

processor = DocumentProcessor(PDF_PATH)

chunks = processor.process()

table_chunks = [
    c for c in processor.get_table_chunks()
    if len(c.table_data) >= 2
]

print(f"\nTable chunks to process: {len(table_chunks)}")

# ------------------------------------------------------------ #
# Preview extracted tables
# ------------------------------------------------------------ #

for chunk in table_chunks[:3]:

    print("\n" + "=" * 60)
    print(f"Chunk: {chunk.chunk_id}")
    print(f"Section: {chunk.section_heading}")
    print("-" * 60)

    print(chunk.content[:500])

# ------------------------------------------------------------ #
# Run Agent 1
# ------------------------------------------------------------ #

agent = ExtractionAgent()

records = agent.run(table_chunks)

# ------------------------------------------------------------ #
# Print sample output
# ------------------------------------------------------------ #

print("\n" + "=" * 60)
print("EXTRACTED RECORDS SAMPLE")
print("=" * 60)

for r in records[:10]:

    print(
        f"[{r.section_heading[:30]}] "
        f"{r.parameter:<30} "
        f"value={r.extracted_value:<10} "
        f"unit={r.unit:<10} "
        f"context={r.row_context}"
    )

# ------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------ #

unique_parameters = len(
    set(r.parameter for r in records)
)

print("\n" + "=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)

print(f"Total records: {len(records)}")
print(f"Unique parameters: {unique_parameters}")
print(f"Processed tables: {len(table_chunks)}")

# ------------------------------------------------------------ #
# Save output
# ------------------------------------------------------------ #

output_path = "output/agent1_output.json"

with open(output_path, "w") as f:

    json.dump(
        [r.model_dump() for r in records],
        f,
        indent=2,
    )

print(
    f"\n✅ Agent 1 done. "
    f"{len(records)} records saved to {output_path}"
)
