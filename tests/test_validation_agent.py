"""
Test Agent 2 in isolation.          
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pipeline.document_processor import DocumentProcessor
from pipeline.vector_store import VectorStore
from agents.extraction_agent import ExtractionAgent
from agents.validation_agent import ValidationAgent

PDF_PATH = "sample_extract_call.pdf"

# Step 1: Ingest + index
print("=" * 60)
print("STEP 1: Ingesting document into vector store")
print("=" * 60)
processor = DocumentProcessor(PDF_PATH)
chunks = processor.process()

store = VectorStore()
store.reset()   # fresh index each run
store.index(chunks)

# Step 2: Run Agent 1
print("\n" + "=" * 60)
print("STEP 2: Agent 1 — Extraction")
print("=" * 60)
agent1 = ExtractionAgent()
records = agent1.run(processor.get_table_chunks())

# Step 3: Run Agent 2
print("\n" + "=" * 60)
print("STEP 3: Agent 2 — Validation")
print("=" * 60)
agent2 = ValidationAgent(vector_store=store, use_rag=True)
validated = agent2.run(records)

# Step 4: Print summary
print("\n" + "=" * 60)
print("VALIDATION REPORT SAMPLE (first 20)")
print("=" * 60)
for r in validated[:20]:
    status_icon = "✅" if r.validation_status == "PASS" else (
                  "❌" if r.validation_status == "FAIL" else "⏭️ ")
    print(f"{status_icon} [{r.validation_status:<7}] "
          f"{r.parameter:<35} "
          f"value={r.extracted_value:<10} "
          f"rule='{r.compliance_range}'")

# Print FAILs separately
fails = [r for r in validated if r.validation_status == "FAIL"]
if fails:
    print(f"\n{'='*60}")
    print(f"FAILURES ({len(fails)} total)")
    print("=" * 60)
    for r in fails:
        print(f"  ❌ {r.parameter} = {r.extracted_value} {r.unit}")
        print(f"     Rule  : {r.compliance_range}")
        print(f"     Note  : {r.validation_note}")
        print(f"     Where : {r.row_context} | {r.section_heading[:50]}")
        print()

# Save output
with open("output/agent2_output.json", "w") as f:
    json.dump([r.model_dump() for r in validated], f, indent=2)

print(f"\n✅ Agent 2 done. {len(validated)} records saved to output/agent2_output.json")
