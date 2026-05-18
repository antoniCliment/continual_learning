#!/usr/bin/env python3
"""
Create SVG plots for GSM8K judge_results CSV files.

The script is intentionally dependency-free: it reads CSVs with the standard
library and writes SVG plots directly. By default it looks for judge result CSVs
in ../results that contain both `correct` and `gemma_judge_correct`.
"""

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BENCHMARK_DIR / "results"
PLOTS_DIR = BENCHMARK_DIR / "plots"

AUTO_COLOR = "#2563eb"
JUDGE_COLOR = "#059669"
AGREE_COLOR = "#7c3aed"
UNKNOWN_COLOR = "#6b7280"
GRID_COLOR = "#d1d5db"
TEXT_COLOR = "#111827"
MUTED_TEXT = "#4b5563"
PANEL_BORDER = "#e5e7eb"

BREAKDOWN_COLORS = {
    "Both correct": "#16a34a",
    "Auto only": "#f59e0b",
    "Judge only": "#0ea5e9",
    "Both incorrect": "#dc2626",
    "Unknown": UNKNOWN_COLOR,
}


@dataclass
class Stats:
    total: int = 0
    auto_known: int = 0
    auto_correct: int = 0
    judge_known: int = 0
    judge_correct: int = 0
    judge_unknown: int = 0
    agreement_known: int = 0
    agreement: int = 0
    both_correct: int = 0
    auto_only_correct: int = 0
    judge_only_correct: int = 0
    both_incorrect: int = 0

    def add(self, auto_correct, judge_correct):
        self.total += 1

        if auto_correct is not None:
            self.auto_known += 1
            self.auto_correct += int(auto_correct)

        if judge_correct is None:
            self.judge_unknown += 1
        else:
            self.judge_known += 1
            self.judge_correct += int(judge_correct)

        if auto_correct is None or judge_correct is None:
            return

        self.agreement_known += 1
        self.agreement += int(auto_correct == judge_correct)

        if auto_correct and judge_correct:
            self.both_correct += 1
        elif auto_correct and not judge_correct:
            self.auto_only_correct += 1
        elif judge_correct and not auto_correct:
            self.judge_only_correct += 1
        else:
            self.both_incorrect += 1

    @property
    def auto_accuracy(self):
        return percent(self.auto_correct, self.auto_known)

    @property
    def judge_accuracy(self):
        return percent(self.judge_correct, self.judge_known)

    @property
    def agreement_rate(self):
        return percent(self.agreement, self.agreement_known)

    @property
    def judge_unknown_rate(self):
        return percent(self.judge_unknown, self.total)


def percent(numerator, denominator):
    if not denominator:
        return None
    return (numerator / denominator) * 100


def parse_bool(value):
    if value is None:
        return None

    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"1", "true", "yes", "correct"}:
        return True
    if text in {"0", "false", "no", "incorrect"}:
        return False

    try:
        number = float(text)
    except ValueError:
        return None

    if math.isnan(number):
        return None
    if number == 1:
        return True
    if number == 0:
        return False
    return None


def safe_label(value, fallback="unknown"):
    text = str(value).strip() if value is not None else ""
    return text or fallback


def short_label(value, max_len=22):
    text = safe_label(value)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}..."


def numeric_sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def step_type(row):
    return "final" if parse_bool(row.get("is_final_step")) else "intermediate"


def has_required_columns(path):
    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except OSError:
        return False
    return {"correct", "gemma_judge_correct"}.issubset(set(header))


def discover_judge_files(results_dir=RESULTS_DIR):
    paths = []
    for pattern in ("judge_results*.csv", "judge_*results*.csv"):
        paths.extend(results_dir.glob(pattern))

    unique_paths = []
    seen = set()
    for path in sorted(paths):
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if has_required_columns(path):
            unique_paths.append(path)
    return unique_paths


def load_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")

        required_columns = {"correct", "gemma_judge_correct"}
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        return list(reader)


def add_to_group(groups, group_name, key, auto_correct, judge_correct):
    groups.setdefault(group_name, {})
    stats = groups[group_name].setdefault(key, Stats())
    stats.add(auto_correct, judge_correct)


def summarize(rows):
    groups = {
        "overall": {"all": Stats()},
        "step_index": {},
        "evaluation_source": {},
        "evaluation_kind": {},
        "step_type": {},
        "total_steps": {},
    }

    for row in rows:
        auto_correct = parse_bool(row.get("correct"))
        judge_correct = parse_bool(row.get("gemma_judge_correct"))

        groups["overall"]["all"].add(auto_correct, judge_correct)
        add_to_group(
            groups,
            "step_index",
            safe_label(row.get("step_index")),
            auto_correct,
            judge_correct,
        )
        add_to_group(
            groups,
            "evaluation_source",
            safe_label(row.get("evaluation_source")),
            auto_correct,
            judge_correct,
        )
        add_to_group(
            groups,
            "evaluation_kind",
            safe_label(row.get("evaluation_kind")),
            auto_correct,
            judge_correct,
        )
        add_to_group(groups, "step_type", step_type(row), auto_correct, judge_correct)
        add_to_group(
            groups,
            "total_steps",
            safe_label(row.get("total_steps")),
            auto_correct,
            judge_correct,
        )

    return groups


def svg_text(x, y, text, size=13, weight="400", color=TEXT_COLOR, anchor="start", extra=""):
    escaped = html.escape(str(text))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" font-weight="{weight}" '
        f'fill="{color}" text-anchor="{anchor}" {extra}>{escaped}</text>'
    )


def draw_panel(parts, x, y, width, height, title):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="8" fill="#ffffff" stroke="{PANEL_BORDER}" />'
    )
    parts.append(svg_text(x + 18, y + 30, title, size=18, weight="700"))


def y_position(value, top, plot_height, y_max=100):
    value = max(0, min(y_max, value))
    return top + plot_height - (value / y_max) * plot_height


def draw_rate_grid(parts, left, top, width, height):
    for tick in (0, 25, 50, 75, 100):
        y = y_position(tick, top, height)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1" />'
        )
        parts.append(svg_text(left - 8, y + 4, tick, size=11, color=MUTED_TEXT, anchor="end"))
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" '
        f'stroke="{TEXT_COLOR}" stroke-width="1" />'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" '
        f'stroke="{TEXT_COLOR}" stroke-width="1" />'
    )


def draw_legend(parts, x, y, entries, columns=3):
    col_width = 150
    row_height = 22
    for idx, (label, color) in enumerate(entries):
        col = idx % columns
        row = idx // columns
        entry_x = x + col * col_width
        entry_y = y + row * row_height
        parts.append(
            f'<rect x="{entry_x}" y="{entry_y - 10}" width="12" height="12" '
            f'fill="{color}" />'
        )
        parts.append(svg_text(entry_x + 18, entry_y + 1, label, size=12, color=MUTED_TEXT))


def draw_overall_bars(parts, x, y, width, height, stats):
    draw_panel(parts, x, y, width, height, "Overall judge comparison")

    chart_left = x + 76
    chart_top = y + 64
    chart_width = width - 112
    chart_height = height - 140
    draw_rate_grid(parts, chart_left, chart_top, chart_width, chart_height)

    values = [
        ("Auto accuracy", stats.auto_accuracy, AUTO_COLOR),
        ("Judge accuracy", stats.judge_accuracy, JUDGE_COLOR),
        ("Agreement", stats.agreement_rate, AGREE_COLOR),
        ("Judge unknown", stats.judge_unknown_rate, UNKNOWN_COLOR),
    ]
    bar_gap = 28
    bar_width = min(84, (chart_width - bar_gap * (len(values) + 1)) / len(values))
    x_cursor = chart_left + bar_gap

    for label, value, color in values:
        value = value or 0
        bar_height = (value / 100) * chart_height
        bar_y = chart_top + chart_height - bar_height
        parts.append(
            f'<rect x="{x_cursor:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}" />'
        )
        parts.append(svg_text(x_cursor + bar_width / 2, bar_y - 8, f"{value:.1f}%", size=12, anchor="middle"))
        parts.append(
            svg_text(
                x_cursor + bar_width / 2,
                chart_top + chart_height + 20,
                short_label(label, 14),
                size=11,
                color=MUTED_TEXT,
                anchor="middle",
            )
        )
        x_cursor += bar_width + bar_gap

    compared = stats.agreement_known
    footer = (
        f"Rows: {stats.total} | judged: {stats.judge_known} | "
        f"unknown: {stats.judge_unknown} | compared: {compared}"
    )
    parts.append(svg_text(x + 18, y + height - 20, footer, size=12, color=MUTED_TEXT))


def point_x(index, count, left, width):
    if count <= 1:
        return left + width / 2
    return left + (index / (count - 1)) * width


def draw_line_chart(parts, x, y, width, height, title, items):
    draw_panel(parts, x, y, width, height, title)
    if not items:
        parts.append(svg_text(x + 18, y + 70, "No data", color=MUTED_TEXT))
        return

    chart_left = x + 64
    chart_top = y + 64
    chart_width = width - 100
    chart_height = height - 130
    draw_rate_grid(parts, chart_left, chart_top, chart_width, chart_height)

    labels = [label for label, _stats in items]
    series = [
        ("Auto", AUTO_COLOR, [stats.auto_accuracy for _label, stats in items]),
        ("Judge", JUDGE_COLOR, [stats.judge_accuracy for _label, stats in items]),
        ("Agreement", AGREE_COLOR, [stats.agreement_rate for _label, stats in items]),
    ]

    for _name, color, values in series:
        points = []
        for idx, value in enumerate(values):
            if value is None:
                continue
            px = point_x(idx, len(values), chart_left, chart_width)
            py = y_position(value, chart_top, chart_height)
            points.append((px, py))
        if not points:
            continue
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" '
            f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />'
        )
        for px, py in points:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" />')

    label_every = max(1, math.ceil(len(labels) / 14))
    for idx, label in enumerate(labels):
        if idx % label_every != 0 and idx != len(labels) - 1:
            continue
        px = point_x(idx, len(labels), chart_left, chart_width)
        parts.append(svg_text(px, chart_top + chart_height + 20, short_label(label, 8), size=11, color=MUTED_TEXT, anchor="middle"))

    draw_legend(
        parts,
        x + 18,
        y + height - 22,
        [("Auto", AUTO_COLOR), ("Judge", JUDGE_COLOR), ("Agreement", AGREE_COLOR)],
    )


def draw_grouped_bar_chart(parts, x, y, width, height, title, items):
    draw_panel(parts, x, y, width, height, title)
    if not items:
        parts.append(svg_text(x + 18, y + 70, "No data", color=MUTED_TEXT))
        return

    chart_left = x + 64
    chart_top = y + 64
    chart_width = width - 100
    chart_height = height - 140
    draw_rate_grid(parts, chart_left, chart_top, chart_width, chart_height)

    series = [
        ("Auto", AUTO_COLOR, lambda stats: stats.auto_accuracy),
        ("Judge", JUDGE_COLOR, lambda stats: stats.judge_accuracy),
        ("Agreement", AGREE_COLOR, lambda stats: stats.agreement_rate),
    ]

    group_width = chart_width / len(items)
    bar_width = min(28, group_width / 5)
    for idx, (label, stats) in enumerate(items):
        center = chart_left + group_width * idx + group_width / 2
        first_x = center - ((len(series) * bar_width) + ((len(series) - 1) * 4)) / 2
        for series_idx, (_name, color, getter) in enumerate(series):
            value = getter(stats) or 0
            bar_height = (value / 100) * chart_height
            bar_x = first_x + series_idx * (bar_width + 4)
            bar_y = chart_top + chart_height - bar_height
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}" />'
            )
        parts.append(
            svg_text(
                center,
                chart_top + chart_height + 18,
                short_label(label, 16),
                size=10,
                color=MUTED_TEXT,
                anchor="middle",
            )
        )

    draw_legend(
        parts,
        x + 18,
        y + height - 24,
        [("Auto", AUTO_COLOR), ("Judge", JUDGE_COLOR), ("Agreement", AGREE_COLOR)],
    )


def draw_stacked_step_chart(parts, x, y, width, height, title, items):
    draw_panel(parts, x, y, width, height, title)
    if not items:
        parts.append(svg_text(x + 18, y + 70, "No data", color=MUTED_TEXT))
        return

    chart_left = x + 64
    chart_top = y + 64
    chart_width = width - 100
    chart_height = height - 150
    draw_rate_grid(parts, chart_left, chart_top, chart_width, chart_height)

    group_width = chart_width / len(items)
    bar_width = min(34, group_width * 0.62)
    categories = [
        "Both correct",
        "Auto only",
        "Judge only",
        "Both incorrect",
        "Unknown",
    ]

    for idx, (label, stats) in enumerate(items):
        bar_x = chart_left + group_width * idx + (group_width - bar_width) / 2
        y_cursor = chart_top + chart_height
        values = {
            "Both correct": stats.both_correct,
            "Auto only": stats.auto_only_correct,
            "Judge only": stats.judge_only_correct,
            "Both incorrect": stats.both_incorrect,
            "Unknown": stats.total - stats.agreement_known,
        }
        for category in categories:
            raw_value = values[category]
            rate = (raw_value / stats.total) * 100 if stats.total else 0
            segment_height = (rate / 100) * chart_height
            y_cursor -= segment_height
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{y_cursor:.1f}" width="{bar_width:.1f}" '
                f'height="{segment_height:.1f}" fill="{BREAKDOWN_COLORS[category]}" />'
            )
        parts.append(svg_text(bar_x + bar_width / 2, chart_top + chart_height + 18, short_label(label, 8), size=10, color=MUTED_TEXT, anchor="middle"))

    draw_legend(
        parts,
        x + 18,
        y + height - 48,
        [(category, BREAKDOWN_COLORS[category]) for category in categories],
        columns=3,
    )


def top_items(group, limit=8):
    return sorted(group.items(), key=lambda item: (-item[1].total, numeric_sort_key(item[0])))[:limit]


def build_svg(path, groups, title):
    width = 1400
    height = 960
    margin = 30
    gap = 24
    panel_width = (width - (2 * margin) - gap) / 2
    panel_height = 405

    step_items = sorted(groups["step_index"].items(), key=lambda item: numeric_sort_key(item[0]))
    source_items = top_items(groups["evaluation_source"], limit=8)
    overall = groups["overall"]["all"]

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#f9fafb" />',
        svg_text(margin, 38, title, size=24, weight="700"),
        svg_text(
            margin,
            62,
            "Auto is the benchmark exact/numeric scorer. Judge is gemma_judge_correct.",
            size=13,
            color=MUTED_TEXT,
        ),
    ]

    top_y = 88
    bottom_y = top_y + panel_height + gap
    draw_overall_bars(parts, margin, top_y, panel_width, panel_height, overall)
    draw_line_chart(
        parts,
        margin + panel_width + gap,
        top_y,
        panel_width,
        panel_height,
        "Accuracy and agreement by step",
        step_items,
    )
    draw_stacked_step_chart(
        parts,
        margin,
        bottom_y,
        panel_width,
        panel_height,
        "Agreement categories by step",
        step_items,
    )
    draw_grouped_bar_chart(
        parts,
        margin + panel_width + gap,
        bottom_y,
        panel_width,
        panel_height,
        "Top evaluation sources",
        source_items,
    )

    parts.append(
        svg_text(
            margin,
            height - 24,
            f"Source: {short_label(path.name, 130)}",
            size=12,
            color=MUTED_TEXT,
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def output_base_name(input_path):
    stem = input_path.stem
    if stem.startswith("judge_results_"):
        stem = stem.removeprefix("judge_results_")
    return f"judge_plots_{stem}"


def process_file(input_path, output_dir=None):
    input_path = input_path.resolve()
    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"{input_path} has no data rows")

    groups = summarize(rows)
    destination = (output_dir or PLOTS_DIR).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    base_name = output_base_name(input_path)
    svg_path = destination / f"{base_name}.svg"

    svg = build_svg(input_path, groups, title="GSM8K judge results")
    svg_path.write_text(svg, encoding="utf-8")

    return svg_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create SVG plots for GSM8K judge_results CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help=(
            "Judge result CSV files. Defaults to judge_results*.csv files in this "
            "benchmark's results directory that contain gemma_judge_correct."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory for generated SVG files. Defaults to ../plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = args.csv_files or discover_judge_files()
    if not input_files:
        raise SystemExit(f"No judge result CSV files found in {RESULTS_DIR}")

    for input_path in input_files:
        svg_path = process_file(
            input_path,
            output_dir=args.output_dir,
        )
        print(f"Created plot: {svg_path}")


if __name__ == "__main__":
    main()
