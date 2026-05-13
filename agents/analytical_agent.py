"""
Agent 3 - Deterministic Analytical Agent

Responsibilities:
- Receives validated records from Agent 2
- Computes statistical summaries (mean, median, mode, std, min, max, Cpk)
- Detects trends across months (linear-regression slope)
- Generates rule-based pharmacological insights
- Produces matplotlib chart files (run chart, pass/fail bar, failure heatmap)
- Returns AnalyticalSummary objects
"""

import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.schemas import AnalyticalSummary, ValidationRecord
from rules.compliance_rules import find_rule


MONTH_PATTERN = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}\b"
)


class AnalyticalAgent:

    def __init__(self, output_dir: str = "output"):

        self.name = "AnalyticalAgent"
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------ #

    def run(
        self,
        records: list[ValidationRecord],
    ) -> list[AnalyticalSummary]:

        # Step 1 - filter to PASS + FAIL only
        usable = self._filter_records(records)

        if not usable:
            print(f"[{self.name}] No usable PASS/FAIL records to analyze.")
            return []

        # Step 2 + 3 - parse context and group
        groups = self._group_records(usable)

        # Step 4-7 - compute and build summaries
        summaries = []

        for (parameter, section), group_records in groups.items():

            summary = self._build_summary(
                parameter,
                section,
                group_records,
            )

            if summary is not None:
                summaries.append(summary)

        # Step 8 - cross-parameter charts
        bar_path = self._chart_pass_fail(summaries)

        heatmap_path = self._chart_failure_heatmap(usable)

        # Attach cross-parameter charts to each summary
        for s in summaries:

            if bar_path:
                s.chart_files.append(bar_path)

            if heatmap_path:
                s.chart_files.append(heatmap_path)

        print(f"\n[{self.name}] Analysis complete:")
        print(f"  Parameters analyzed : {len(summaries)}")
        print(f"  Charts directory    : {self.charts_dir}")

        return summaries

    # ------------------------------------------------------------ #
    # Filtering and grouping
    # ------------------------------------------------------------ #

    def _filter_records(
        self,
        records: list[ValidationRecord],
    ) -> list[ValidationRecord]:

        usable = []

        for r in records:

            if r.validation_status not in ("PASS", "FAIL"):
                continue

            # numeric values only
            try:
                float(r.extracted_value)
            except (TypeError, ValueError):
                continue

            usable.append(r)

        return usable

    def _parse_context(
        self,
        row_context: str | None,
    ) -> tuple[str | None, str | None]:

        if not row_context:
            return None, None

        month_match = MONTH_PATTERN.search(row_context)
        month = month_match.group(0) if month_match else None

        # strip month and separators to recover area
        remainder = row_context

        if month:
            remainder = remainder.replace(month, "")

        remainder = remainder.strip(" |:-\t")

        # numeric-only remainders are not areas (they are min/max observations)
        if remainder and not re.fullmatch(r"[\d.\s]+", remainder):
            area = remainder
        else:
            area = None

        return month, area

    def _group_records(
        self,
        records: list[ValidationRecord],
    ) -> dict[tuple[str, str], list[dict]]:

        groups = defaultdict(list)

        for r in records:

            month, area = self._parse_context(r.row_context)

            groups[(r.parameter, r.section_heading)].append(
                {
                    "value": float(r.extracted_value),
                    "status": r.validation_status,
                    "unit": r.unit,
                    "compliance_range": r.compliance_range,
                    "month": month,
                    "area": area,
                }
            )

        return groups

    # ------------------------------------------------------------ #
    # Per-parameter summary
    # ------------------------------------------------------------ #

    def _build_summary(
        self,
        parameter: str,
        section: str,
        group: list[dict],
    ) -> AnalyticalSummary | None:

        if not group:
            return None

        values = [g["value"] for g in group]

        n = len(values)

        # ------- core stats -------
        mean = float(np.mean(values))
        median = float(np.median(values))

        try:
            mode = float(statistics.mode(values))
        except statistics.StatisticsError:
            mode = None

        std_dev = float(np.std(values, ddof=1)) if n > 1 else 0.0
        min_value = float(np.min(values))
        max_value = float(np.max(values))

        pass_count = sum(1 for g in group if g["status"] == "PASS")
        fail_count = sum(1 for g in group if g["status"] == "FAIL")

        # ------- Cpk -------
        cpk = self._compute_cpk(parameter, section, mean, std_dev)

        # ------- trend -------
        trend = self._detect_trend(group, mean)

        # ------- insight -------
        cv = (std_dev / abs(mean)) if mean else 0.0

        insight = self._generate_insight(
            parameter=parameter,
            count=n,
            fail_count=fail_count,
            cpk=cpk,
            trend=trend,
            cv=cv,
        )

        # ------- per-parameter run chart -------
        chart_files = []

        run_chart = self._chart_run(parameter, section, group)

        if run_chart:
            chart_files.append(run_chart)

        unit = group[0].get("unit") or None

        return AnalyticalSummary(
            parameter=parameter,
            section=section,
            unit=unit,
            count=n,
            mean=round(mean, 4),
            median=round(median, 4),
            mode=(round(mode, 4) if mode is not None else None),
            std_dev=round(std_dev, 4),
            min_value=round(min_value, 4),
            max_value=round(max_value, 4),
            cpk=(round(cpk, 3) if cpk is not None else None),
            trend=trend,
            pass_count=pass_count,
            fail_count=fail_count,
            insight=insight,
            chart_files=chart_files,
        )

    # ------------------------------------------------------------ #
    # Cpk
    # ------------------------------------------------------------ #

    def _compute_cpk(
        self,
        parameter: str,
        section: str,
        mean: float,
        std_dev: float,
    ) -> float | None:

        if std_dev == 0:
            return None

        rule = find_rule(section, parameter)

        if rule is None:
            return None

        usl = rule.get("max")
        lsl = rule.get("min")

        if usl is None and lsl is None:
            return None

        cpu = (usl - mean) / (3 * std_dev) if usl is not None else float("inf")
        cpl = (mean - lsl) / (3 * std_dev) if lsl is not None else float("inf")

        cpk = min(cpu, cpl)

        if cpk == float("inf"):
            return None

        return float(cpk)

    # ------------------------------------------------------------ #
    # Trend detection
    # ------------------------------------------------------------ #

    def _detect_trend(
        self,
        group: list[dict],
        mean: float,
    ) -> str:

        # only entries with a parseable month contribute to a trend
        time_pairs = []

        for g in group:

            if not g["month"]:
                continue

            try:
                dt = datetime.strptime(g["month"], "%b-%y")
            except ValueError:
                continue

            time_pairs.append((dt, g["value"]))

        if len(time_pairs) < 3:
            return "INSUFFICIENT_DATA"

        time_pairs.sort(key=lambda p: p[0])

        xs = np.arange(len(time_pairs))
        ys = np.array([v for _, v in time_pairs], dtype=float)

        slope, _ = np.polyfit(xs, ys, 1)

        threshold = 0.05 * abs(mean) if mean else 0.05

        if abs(slope) < threshold:
            return "STABLE"

        return "INCREASING" if slope > 0 else "DECREASING"

    # ------------------------------------------------------------ #
    # Insight generation
    # ------------------------------------------------------------ #

    def _generate_insight(
        self,
        parameter: str,
        count: int,
        fail_count: int,
        cpk: float | None,
        trend: str,
        cv: float,
    ) -> str:

        fail_ratio = fail_count / count if count else 0.0

        if fail_count > 0 and fail_ratio > 0.30:
            return (
                f"{parameter} had {fail_count}/{count} failures - "
                f"high non-compliance rate detected."
            )

        if cpk is not None and cpk < 1.0:
            return (
                f"Process capability below threshold (Cpk={cpk:.2f}). "
                f"{parameter} is unstable - investigation recommended."
            )

        if trend == "INCREASING" and parameter.endswith("_max"):
            return (
                f"{parameter} shows an increasing trend - "
                f"risk of breaching the upper compliance limit."
            )

        if trend == "DECREASING" and parameter.endswith("_min"):
            return (
                f"{parameter} shows a decreasing trend - "
                f"risk of dropping below the minimum limit."
            )

        if cv > 0.15:
            return (
                f"{parameter} shows high variability (CV={cv*100:.1f}%). "
                f"Root cause investigation recommended."
            )

        return (
            f"{parameter} remained stable across all observations "
            f"and within the compliance range."
        )

    # ------------------------------------------------------------ #
    # Charts
    # ------------------------------------------------------------ #

    def _safe_filename(self, parameter: str) -> str:

        return re.sub(r"[^a-z0-9_]+", "_", parameter.lower()).strip("_")

    def _chart_run(
        self,
        parameter: str,
        section: str,
        group: list[dict],
    ) -> str | None:

        # collect time-ordered series; skip if not enough temporal data
        time_pairs = []

        for g in group:

            if not g["month"]:
                continue

            try:
                dt = datetime.strptime(g["month"], "%b-%y")
            except ValueError:
                continue

            time_pairs.append((dt, g["value"], g["status"]))

        if len(time_pairs) < 2:
            return None

        time_pairs.sort(key=lambda p: p[0])

        xs = [p[0] for p in time_pairs]
        ys = [p[1] for p in time_pairs]

        # compliance limit from rule
        rule = find_rule(section, parameter)
        usl = rule.get("max") if rule else None
        lsl = rule.get("min") if rule else None

        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.6, label=parameter)

        # mark failures in red
        fail_xs = [p[0] for p in time_pairs if p[2] == "FAIL"]
        fail_ys = [p[1] for p in time_pairs if p[2] == "FAIL"]

        if fail_xs:
            ax.scatter(fail_xs, fail_ys, color="#d62728", s=60, zorder=5, label="FAIL")

        if usl is not None:
            ax.axhline(usl, color="#d62728", linestyle="--", linewidth=1, label=f"USL={usl}")

        if lsl is not None:
            ax.axhline(lsl, color="#2ca02c", linestyle="--", linewidth=1, label=f"LSL={lsl}")

        ax.set_title(f"Run chart - {parameter}\n{section}", fontsize=10)
        ax.set_xlabel("Month")
        ax.set_ylabel(f"Value ({group[0].get('unit') or ''})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()

        path = self.charts_dir / f"run_chart_{self._safe_filename(parameter)}.png"

        fig.savefig(path, dpi=150)
        plt.close(fig)

        return str(path)

    def _chart_pass_fail(
        self,
        summaries: list[AnalyticalSummary],
    ) -> str | None:

        if not summaries:
            return None

        labels = [s.parameter for s in summaries]
        passes = [s.pass_count for s in summaries]
        fails = [s.fail_count for s in summaries]

        x = np.arange(len(labels))
        width = 0.4

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))

        ax.bar(x - width / 2, passes, width, color="#2ca02c", label="PASS")
        ax.bar(x + width / 2, fails, width, color="#d62728", label="FAIL")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title("Pass / Fail counts per parameter")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()

        path = self.charts_dir / "bar_chart_pass_fail.png"

        fig.savefig(path, dpi=150)
        plt.close(fig)

        return str(path)

    def _chart_failure_heatmap(
        self,
        records: list[ValidationRecord],
    ) -> str | None:

        # collect area x parameter fail counts
        matrix = defaultdict(lambda: defaultdict(int))
        areas = set()
        parameters = set()

        for r in records:

            _, area = self._parse_context(r.row_context)

            if not area:
                continue

            if r.validation_status != "FAIL":
                continue

            matrix[area][r.parameter] += 1
            areas.add(area)
            parameters.add(r.parameter)

        if not matrix:
            return None

        area_list = sorted(areas)
        param_list = sorted(parameters)

        data = np.zeros((len(area_list), len(param_list)), dtype=int)

        for i, a in enumerate(area_list):
            for j, p in enumerate(param_list):
                data[i, j] = matrix[a].get(p, 0)

        fig, ax = plt.subplots(
            figsize=(max(6, len(param_list) * 1.2), max(4, len(area_list) * 0.45))
        )

        im = ax.imshow(data, cmap="Reds", aspect="auto")

        ax.set_xticks(np.arange(len(param_list)))
        ax.set_xticklabels(param_list, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(area_list)))
        ax.set_yticklabels(area_list, fontsize=8)

        # annotate cells
        for i in range(len(area_list)):
            for j in range(len(param_list)):
                if data[i, j] > 0:
                    ax.text(j, i, str(data[i, j]), ha="center", va="center", fontsize=8, color="black")

        ax.set_title("Failure heatmap - area x parameter")
        fig.colorbar(im, ax=ax, label="Fail count")
        fig.tight_layout()

        path = self.charts_dir / "heatmap_failures.png"

        fig.savefig(path, dpi=150)
        plt.close(fig)

        return str(path)
