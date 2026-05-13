"""
Test Agent 1 in isolation.
Usage: python test_agent1.py
"""

import json
from pipeline.document_processor import DocumentProcessor
from agents.extraction_agent import ExtractionAgent

PDF_PATH = "sample_extract_call.pdf"

# Ingest document
processor = DocumentProcessor(PDF_PATH)
chunks = processor.process()
table_chunks = processor.get_table_chunks()

print(f"Table chunks to process: {len(table_chunks)}\n")

# Run Agent 1
agent = ExtractionAgent()
records = agent.run(table_chunks)

# Print results
print("\n" + "=" * 60)
print("EXTRACTED RECORDS SAMPLE (first 10)")
print("=" * 60)
for r in records[:10]:
    print(f"  [{r.section_heading[:40]}]"
          f"  {r.parameter:<35}"
          f"  value={r.extracted_value:<10}"
          f"  unit={r.unit:<10}"
          f"  ctx={r.row_context}")

# Save full output
with open("output/agent1_output.json", "w") as f:
    json.dump([r.model_dump() for r in records], f, indent=2)

print(f"\n✅ Agent 1 done. {len(records)} records saved to output/agent1_output.json")
