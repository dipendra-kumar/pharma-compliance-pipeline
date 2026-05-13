"""
Test Agent 3 in isolation against existing agent2_output.json.

Usage:
    python tests/test_analytical_agent.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.analytical_agent import AnalyticalAgent
from models.schemas import ValidationRecord


INPUT_PATH = "output/agent2_output.json"
OUTPUT_PATH = "output/agent3_output.json"


def main():

    if not os.path.exists(INPUT_PATH):
        print(f"Missing {INPUT_PATH}. Run main.py or Agent 2 first.")
        sys.exit(1)

    with open(INPUT_PATH) as f:
        raw = json.load(f)

    records = [ValidationRecord(**r) for r in raw]

    print(f"Loaded {len(records)} validation records from {INPUT_PATH}")

    agent = AnalyticalAgent(output_dir="output")
    summaries = agent.run(records)

    # Save Agent 3 output
    with open(OUTPUT_PATH, "w") as f:
        json.dump([s.model_dump() for s in summaries], f, indent=2)

    print("\n" + "=" * 60)
    print("ANALYTICAL SUMMARIES")
    print("=" * 60)

    for s in summaries:
        print(
            f"\n[{s.section[:40]}] {s.parameter} "
            f"(n={s.count}, pass={s.pass_count}, fail={s.fail_count})"
        )
        print(
            f"  mean={s.mean}  median={s.median}  mode={s.mode}  "
            f"std={s.std_dev}  min={s.min_value}  max={s.max_value}"
        )
        print(f"  cpk={s.cpk}  trend={s.trend}")
        print(f"  insight: {s.insight}")
        if s.chart_files:
            print(f"  charts : {s.chart_files}")

    print(f"\nWrote {len(summaries)} summaries to {OUTPUT_PATH}")

    # Basic sanity assertions
    assert len(summaries) > 0, "Expected at least one analytical summary"

    for s in summaries:
        assert s.count > 0
        assert s.pass_count + s.fail_count == s.count
        assert s.min_value <= s.mean <= s.max_value or s.count == 1

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
