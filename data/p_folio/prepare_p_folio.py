#!/usr/bin/env python3
"""Convert the P-FOLIO spreadsheet CSV into SFT-ready files.

P-FOLIO.csv is not one training example per CSV row. It is an exported
spreadsheet where a row with a non-empty story_id starts a proof block, the
following rows are proof steps, and blank rows are separators/padding.

The companion FOLIO files contain the original premises and conclusions. If
FOLIO.csv or folio*.jsonl files are available in this folder, this script uses
them to build full reasoning prompts. Without them, the output is still useful
as a proof-style corpus, but it does not contain enough information for full
premise-to-conclusion training.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABEL_MAP = {"T": "True", "F": "False", "U": "Uncertain"}
LABEL_REVERSE = {"true": "T", "false": "F", "uncertain": "U", "t": "T", "f": "F", "u": "U"}
STEP_COLUMNS = ("Premises used", "Derivation", "Derivation - Corrected", "Derivation index", "Inference rule")
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)


@dataclass
class ProofStep:
    index: str
    premises_used: str
    derivation: str
    inference_rule: str
    source_line: int
    used_corrected_derivation: bool


@dataclass
class ProofBlock:
    source_line: int
    raw_story_id: str
    story_id: int
    raw_truth_value: str
    truth_value: str
    rows: list[tuple[int, dict[str, str]]]
    occurrence_in_story: int = 0


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_story_id(raw_story_id: str) -> int | None:
    """Accept clean ids and spreadsheet notes like '25(XOR)'."""
    match = re.match(r"^\s*(\d+)", raw_story_id or "")
    return int(match.group(1)) if match else None


def normalize_truth_value(raw_truth_value: str, *, coerce_commented: bool) -> str | None:
    text = clean(raw_truth_value)
    if text in LABEL_MAP:
        return text
    if coerce_commented:
        match = re.match(r"^\s*([TFU])\b", text)
        if match:
            return match.group(1)
    return None


def normalize_folio_label(raw_label: str) -> str | None:
    return LABEL_REVERSE.get(clean(raw_label).lower())


def normalize_split(raw_split: str) -> str | None:
    value = clean(raw_split).lower()
    if value in {"train", "training"}:
        return "train"
    if value in {"dev", "valid", "validation", "val"}:
        return "validation"
    if value in {"test", "testing"}:
        return "test"
    return None


def is_blank_csv_row(row: dict[str, str]) -> bool:
    return not any(clean(value) for value in row.values())


def iter_proof_blocks(
    p_folio_csv: Path,
    *,
    coerce_commented_labels: bool,
) -> tuple[list[ProofBlock], dict[str, Any]]:
    stats: dict[str, Any] = {
        "csv_rows": 0,
        "blank_rows": 0,
        "raw_blocks": 0,
        "skipped_blocks": [],
    }
    blocks: list[ProofBlock] = []
    current: dict[str, Any] | None = None

    with p_folio_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"story_id", "Truth Value", *STEP_COLUMNS}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{p_folio_csv} is missing required columns: {', '.join(missing)}")

        for source_line, row in enumerate(reader, start=2):
            stats["csv_rows"] += 1
            if is_blank_csv_row(row):
                stats["blank_rows"] += 1

            raw_story_id = clean(row.get("story_id"))
            if raw_story_id:
                if current is not None:
                    blocks.append(current["block"])

                story_id = normalize_story_id(raw_story_id)
                raw_truth_value = clean(row.get("Truth Value"))
                truth_value = normalize_truth_value(
                    raw_truth_value,
                    coerce_commented=coerce_commented_labels,
                )
                stats["raw_blocks"] += 1

                if story_id is None or truth_value is None:
                    stats["skipped_blocks"].append(
                        {
                            "source_line": source_line,
                            "story_id": raw_story_id,
                            "truth_value": raw_truth_value,
                            "reason": "invalid story_id or truth value",
                        }
                    )
                    current = None
                    continue

                current = {
                    "block": ProofBlock(
                        source_line=source_line,
                        raw_story_id=raw_story_id,
                        story_id=story_id,
                        raw_truth_value=raw_truth_value,
                        truth_value=truth_value,
                        rows=[],
                    )
                }

            if current is not None:
                current["block"].rows.append((source_line, row))

    if current is not None:
        blocks.append(current["block"])

    occurrence_counts: dict[int, int] = defaultdict(int)
    for block in blocks:
        block.occurrence_in_story = occurrence_counts[block.story_id]
        occurrence_counts[block.story_id] += 1

    stats["kept_blocks"] = len(blocks)
    stats["story_count"] = len(occurrence_counts)
    stats["raw_truth_value_counts"] = dict(Counter(block.raw_truth_value for block in blocks))
    stats["truth_value_counts"] = dict(Counter(block.truth_value for block in blocks))
    return blocks, stats


def extract_steps(block: ProofBlock, *, use_corrected: bool) -> tuple[list[ProofStep], int]:
    steps: list[ProofStep] = []
    incomplete_step_rows = 0

    for source_line, row in block.rows:
        has_step_fields = any(clean(row.get(column)) for column in STEP_COLUMNS)
        if not has_step_fields:
            continue

        corrected = clean(row.get("Derivation - Corrected"))
        original = clean(row.get("Derivation"))
        derivation = corrected if use_corrected and corrected else original
        used_corrected = bool(use_corrected and corrected)

        if not derivation:
            incomplete_step_rows += 1
            continue

        step_index = clean(row.get("Derivation index")) or f"D{len(steps) + 1}"
        steps.append(
            ProofStep(
                index=step_index,
                premises_used=clean(row.get("Premises used")),
                derivation=derivation,
                inference_rule=clean(row.get("Inference rule")) or "NA",
                source_line=source_line,
                used_corrected_derivation=used_corrected,
            )
        )

    return steps, incomplete_step_rows


def normalized_field_map(fieldnames: list[str]) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]+", "", field.lower()): field for field in fieldnames}


def find_field(field_map: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in field_map:
            return field_map[key]
    return None


def parse_premises(raw_premises: str) -> list[str]:
    text = clean(raw_premises)
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [clean(item) for item in parsed if clean(item)]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines if len(lines) > 1 else [text]


def split_multiline_cell(raw_value: str) -> list[str]:
    return [line.strip() for line in clean(raw_value).splitlines() if line.strip()]


def load_folio_context(folio_csv: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {"rows": 0, "matched_rows": 0, "expanded_rows": 0, "warnings": []}
    context_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    occurrence_counts: dict[int, int] = defaultdict(int)
    synthetic_story_ids: dict[str, int] = {}

    with folio_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        field_map = normalized_field_map(fieldnames)

        story_id_field = find_field(field_map, ["story_id", "story id", "storyid"])
        if story_id_field is None and fieldnames and fieldnames[0] == "":
            story_id_field = fieldnames[0]
        premises_field = find_field(
            field_map,
            ["premises", "premises nl", "premises - nl", "nl premises", "natural language premises"],
        )
        conclusion_field = find_field(
            field_map,
            ["conclusion", "conclusions", "conclusions nl", "conclusions - nl", "nl conclusion", "natural language conclusion"],
        )
        conclusion_fol_field = find_field(
            field_map,
            ["conclusion fol", "conclusions fol", "conclusions - fol", "fol conclusion"],
        )
        label_field = find_field(field_map, ["label", "truth value", "truth values", "answer"])
        split_field = find_field(field_map, ["split", "dataset split"])

        if not premises_field or not conclusion_field:
            raise ValueError(
                f"{folio_csv} must contain premises and conclusion columns. "
                f"Found columns: {', '.join(fieldnames)}"
            )

        for row in reader:
            stats["rows"] += 1
            premises = parse_premises(row.get(premises_field, ""))
            premise_key = "\n".join(premises)

            if story_id_field:
                story_id = normalize_story_id(row.get(story_id_field, ""))
            else:
                if premise_key not in synthetic_story_ids:
                    synthetic_story_ids[premise_key] = len(synthetic_story_ids)
                story_id = synthetic_story_ids[premise_key]

            if story_id is None:
                stats["warnings"].append({"row": stats["rows"], "reason": "missing story id"})
                continue

            conclusions = split_multiline_cell(row.get(conclusion_field, ""))
            labels = split_multiline_cell(row.get(label_field, "")) if label_field else []
            conclusion_fols = split_multiline_cell(row.get(conclusion_fol_field, "")) if conclusion_fol_field else []
            split = normalize_split(row.get(split_field, "")) if split_field else None

            if labels or len(conclusions) > 1:
                conclusion_values = conclusions
                conclusion_type = "nl"
                if len(conclusions) != len(labels):
                    if conclusion_fols and len(conclusion_fols) == len(labels):
                        conclusion_values = conclusion_fols
                        conclusion_type = "fol"
                        stats["warnings"].append(
                            {
                                "row": stats["rows"],
                                "story_id": story_id,
                                "reason": "conclusion/truth-value count mismatch; used FOL conclusions",
                                "num_conclusions": len(conclusions),
                                "num_truth_values": len(labels),
                                "num_fol_conclusions": len(conclusion_fols),
                            }
                        )
                    else:
                        stats["warnings"].append(
                            {
                                "row": stats["rows"],
                                "story_id": story_id,
                                "reason": "conclusion/truth-value count mismatch; truncating to shortest count",
                                "num_conclusions": len(conclusions),
                                "num_truth_values": len(labels),
                            }
                        )

                for occurrence, (conclusion, label) in enumerate(zip(conclusion_values, labels)):
                    key = (story_id, occurrence)
                    context_by_key[key] = {
                        "story_id": story_id,
                        "occurrence_in_story": occurrence,
                        "premises": premises,
                        "conclusion": conclusion,
                        "conclusion_type": conclusion_type,
                        "label": normalize_folio_label(label),
                        "split": split,
                        "example_id": None,
                    }
                    stats["matched_rows"] += 1
                    stats["expanded_rows"] += 1
                occurrence_counts[story_id] = max(occurrence_counts[story_id], len(labels))
                continue

            occurrence = occurrence_counts[story_id]
            occurrence_counts[story_id] += 1
            key = (story_id, occurrence)
            context_by_key[key] = {
                "story_id": story_id,
                "occurrence_in_story": occurrence,
                "premises": premises,
                "conclusion": clean(row.get(conclusion_field)),
                "conclusion_type": "nl",
                "label": None,
                "split": split,
                "example_id": None,
            }
            stats["matched_rows"] += 1
            stats["expanded_rows"] += 1

    stats["story_count"] = len(occurrence_counts)
    return context_by_key, stats


def apply_jsonl_splits(
    context_by_key: dict[tuple[int, int], dict[str, Any]],
    folio_jsonl_paths: list[Path],
) -> dict[str, Any]:
    split_context, split_stats = load_folio_jsonl_context(folio_jsonl_paths)
    matched = 0
    for key, split_record in split_context.items():
        if key not in context_by_key:
            continue
        context_by_key[key]["split"] = split_record.get("split")
        context_by_key[key]["example_id"] = split_record.get("example_id")
        matched += 1
    split_stats["splits_applied_to_csv_context"] = matched
    return split_stats


def split_from_filename(path: Path) -> str | None:
    stem = path.stem.lower()
    if "train" in stem:
        return "train"
    if any(name in stem for name in ("validation", "valid", "val", "dev")):
        return "validation"
    if "test" in stem:
        return "test"
    return None


def load_folio_jsonl_context(
    folio_jsonl_paths: list[Path],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {"files": [], "rows": 0, "matched_rows": 0, "warnings": []}
    context_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    occurrence_counts: dict[int, int] = defaultdict(int)

    for path in folio_jsonl_paths:
        file_split = split_from_filename(path)
        file_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                stats["rows"] += 1
                file_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats["warnings"].append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "reason": f"invalid JSON: {exc}",
                        }
                    )
                    continue

                story_id = normalize_story_id(str(row.get("story_id", "")))
                if story_id is None:
                    stats["warnings"].append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "reason": "missing story_id",
                        }
                    )
                    continue

                occurrence = occurrence_counts[story_id]
                occurrence_counts[story_id] += 1
                key = (story_id, occurrence)
                context_by_key[key] = {
                    "story_id": story_id,
                    "occurrence_in_story": occurrence,
                    "premises": parse_premises(row.get("premises", "")),
                    "conclusion": clean(row.get("conclusion")),
                    "label": normalize_folio_label(row.get("label", "")),
                    "split": normalize_split(row.get("split", "")) or file_split,
                    "example_id": row.get("example_id"),
                    "source_file": str(path),
                    "source_line": line_number,
                }
                stats["matched_rows"] += 1

        stats["files"].append({"path": str(path), "rows": file_rows, "split": file_split})

    stats["story_count"] = len(occurrence_counts)
    return context_by_key, stats


def build_full_reasoning_prompt(context: dict[str, Any]) -> str:
    premises = context.get("premises") or []
    premise_lines = "\n".join(f"{index}. {premise}" for index, premise in enumerate(premises, start=1))
    conclusion = context.get("conclusion") or ""
    conclusion_heading = "Conclusion (FOL)" if context.get("conclusion_type") == "fol" else "Conclusion"
    return (
        "Using deductive reasoning, determine whether the conclusion is True, False, "
        'or Uncertain based only on the premises. Show the reasoning process, then finish with "Truth value:".\n\n'
        f"Premises:\n{premise_lines}\n\n"
        f"{conclusion_heading}:\n{conclusion}"
    )


def build_proof_only_prompt(block: ProofBlock) -> str:
    return (
        "Write the annotated P-FOLIO proof for this example. The original FOLIO premises "
        "and conclusion were not found, so this record is proof-style data rather than a "
        "complete premise-to-conclusion reasoning example.\n\n"
        f"Story id: {block.story_id}\n"
        f"Conclusion occurrence in story: {block.occurrence_in_story}\n"
        f"Annotated truth value: {LABEL_MAP[block.truth_value]}"
    )


def build_output(steps: list[ProofStep], truth_value: str) -> str:
    lines = ["Reasoning process:"]
    if steps:
        for step in steps:
            source = f"From {step.premises_used}" if step.premises_used else "From the available premises"
            lines.append(f"{step.index}. {source}, by {step.inference_rule}: {step.derivation}")
    elif truth_value == "U":
        lines.append("No conclusive derivation is annotated for this example.")
    else:
        lines.append("No derivation steps are annotated for this example.")

    lines.append(f"Truth value: {LABEL_MAP[truth_value]}")
    return "\n".join(lines)


def split_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
    split_by_story: bool,
) -> None:
    records_with_folio_split = [record for record in records if record["metadata"].get("folio_split")]
    if records_with_folio_split:
        for record in records:
            if record["metadata"].get("folio_label_mismatch"):
                record["split"] = "test"
            else:
                record["split"] = record["metadata"].get("folio_split") or "test"
        return

    random_instance = random.Random(seed)
    train_ratio, validation_ratio, _ = ratios

    if split_by_story:
        groups = sorted({record["metadata"]["story_id"] for record in records})
        random_instance.shuffle(groups)
        n_train = int(len(groups) * train_ratio)
        n_validation = int(len(groups) * validation_ratio)
        train_ids = set(groups[:n_train])
        validation_ids = set(groups[n_train : n_train + n_validation])

        for record in records:
            story_id = record["metadata"]["story_id"]
            if story_id in train_ids:
                record["split"] = "train"
            elif story_id in validation_ids:
                record["split"] = "validation"
            else:
                record["split"] = "test"
        return

    shuffled = records[:]
    random_instance.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_validation = int(len(shuffled) * validation_ratio)
    for index, record in enumerate(shuffled):
        if index < n_train:
            record["split"] = "train"
        elif index < n_train + n_validation:
            record["split"] = "validation"
        else:
            record["split"] = "test"


def parse_split_ratios(raw_ratios: str) -> tuple[float, float, float]:
    parts = [float(part) for part in raw_ratios.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split ratios must be three comma-separated numbers")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("split ratios must sum to a positive number")
    return tuple(part / total for part in parts)  # type: ignore[return-value]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "messages": record["messages"],
                        "metadata": record["metadata"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "input",
        "output",
        "text",
        "sample_id",
        "story_id",
        "occurrence_in_story",
        "truth_value",
        "label",
        "num_steps",
        "source_line",
        "has_folio_context",
        "folio_label",
        "folio_label_mismatch",
        "folio_example_id",
        "conclusion_type",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metadata = record["metadata"]
            user = record["messages"][0]["content"]
            assistant = record["messages"][1]["content"]
            writer.writerow(
                {
                    "input": user,
                    "output": assistant,
                    "text": f"### User:\n{user}\n\n### Assistant:\n{assistant}",
                    "sample_id": metadata["sample_id"],
                    "story_id": metadata["story_id"],
                    "occurrence_in_story": metadata["occurrence_in_story"],
                    "truth_value": metadata["truth_value"],
                    "label": metadata["label"],
                    "num_steps": metadata["num_steps"],
                    "source_line": metadata["source_line"],
                    "has_folio_context": metadata["has_folio_context"],
                    "folio_label": metadata["folio_label"],
                    "folio_label_mismatch": metadata["folio_label_mismatch"],
                    "folio_example_id": metadata["folio_example_id"],
                    "conclusion_type": metadata["conclusion_type"],
                    "split": record["split"],
                }
            )


def build_records(
    blocks: list[ProofBlock],
    *,
    folio_context: dict[tuple[int, int], dict[str, Any]] | None,
    use_corrected: bool,
    drop_empty_proofs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "empty_proof_blocks": 0,
        "dropped_empty_proof_blocks": 0,
        "incomplete_step_rows": 0,
        "corrected_steps_used": 0,
        "missing_folio_context": 0,
        "folio_label_mismatches": [],
    }

    for block in blocks:
        steps, incomplete_step_rows = extract_steps(block, use_corrected=use_corrected)
        stats["incomplete_step_rows"] += incomplete_step_rows
        stats["corrected_steps_used"] += sum(1 for step in steps if step.used_corrected_derivation)

        if not steps:
            stats["empty_proof_blocks"] += 1
            if drop_empty_proofs:
                stats["dropped_empty_proof_blocks"] += 1
                continue

        context = folio_context.get((block.story_id, block.occurrence_in_story)) if folio_context else None
        if context is None:
            stats["missing_folio_context"] += 1
            user_prompt = build_proof_only_prompt(block)
            folio_split = None
            folio_label = None
            folio_label_mismatch = False
        else:
            user_prompt = build_full_reasoning_prompt(context)
            folio_split = context.get("split")
            folio_label = context.get("label")
            folio_label_mismatch = bool(folio_label and folio_label != block.truth_value)
            if folio_label_mismatch:
                stats["folio_label_mismatches"].append(
                    {
                        "source_line": block.source_line,
                        "story_id": block.story_id,
                        "occurrence_in_story": block.occurrence_in_story,
                        "p_folio_label": block.truth_value,
                        "folio_label": folio_label,
                    }
                )

        assistant_output = build_output(steps, block.truth_value)
        sample_id = len(records)
        metadata = {
            "sample_id": sample_id,
            "story_id": block.story_id,
            "raw_story_id": block.raw_story_id,
            "occurrence_in_story": block.occurrence_in_story,
            "truth_value": block.truth_value,
            "label": LABEL_MAP[block.truth_value],
            "num_steps": len(steps),
            "source_line": block.source_line,
            "has_folio_context": context is not None,
            "folio_split": folio_split,
            "folio_example_id": context.get("example_id") if context else None,
            "folio_label": folio_label,
            "folio_label_mismatch": folio_label_mismatch,
            "conclusion_type": context.get("conclusion_type") if context else None,
        }
        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_output},
                ],
                "metadata": metadata,
                "split": None,
            }
        )

    stats["records"] = len(records)
    stats["label_counts"] = dict(Counter(record["metadata"]["truth_value"] for record in records))
    stats["step_count_distribution"] = dict(Counter(record["metadata"]["num_steps"] for record in records))
    return records, stats


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-folio-csv", type=Path, default=script_dir / "P-FOLIO.csv")
    parser.add_argument("--folio-csv", type=Path, default=script_dir / "FOLIO.csv")
    parser.add_argument(
        "--folio-jsonl",
        type=Path,
        nargs="*",
        default=None,
        help="FOLIO JSONL files. Defaults to auto-discovering folio*.jsonl next to this script.",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "processed")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--split-ratios", type=parse_split_ratios, default=DEFAULT_SPLIT_RATIOS)
    parser.add_argument("--split-by-sample", action="store_true", help="Split examples directly instead of grouping by story_id.")
    parser.add_argument("--drop-empty-proofs", action="store_true", help="Drop blocks with zero proof steps.")
    parser.add_argument("--no-corrected", action="store_true", help="Use the original Derivation column even when corrected text exists.")
    parser.add_argument(
        "--coerce-commented-labels",
        action="store_true",
        help="Keep rows whose Truth Value starts with T/F/U but contains comments. Off by default to match the 1,430 clean proofs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.p_folio_csv.exists():
        raise FileNotFoundError(args.p_folio_csv)

    blocks, parse_stats = iter_proof_blocks(
        args.p_folio_csv,
        coerce_commented_labels=args.coerce_commented_labels,
    )

    folio_context = None
    folio_stats = None
    folio_jsonl_paths = (
        sorted(path for path in args.p_folio_csv.parent.glob("folio*.jsonl") if path.is_file())
        if args.folio_jsonl is None
        else args.folio_jsonl
    )
    if args.folio_csv.exists():
        folio_context, folio_stats = load_folio_context(args.folio_csv)
        if folio_jsonl_paths:
            folio_stats["jsonl_split_stats"] = apply_jsonl_splits(folio_context, folio_jsonl_paths)
    elif folio_jsonl_paths:
        folio_context, folio_stats = load_folio_jsonl_context(folio_jsonl_paths)

    records, build_stats = build_records(
        blocks,
        folio_context=folio_context,
        use_corrected=not args.no_corrected,
        drop_empty_proofs=args.drop_empty_proofs,
    )
    split_records(
        records,
        seed=args.seed,
        ratios=args.split_ratios,
        split_by_story=not args.split_by_sample,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_split[record["split"]].append(record)

    write_jsonl(args.output_dir / "all_data.jsonl", records)
    write_csv(args.output_dir / "all_data.csv", records)
    for split_name in ("train", "validation", "test"):
        split_records_list = records_by_split.get(split_name, [])
        write_jsonl(args.output_dir / f"{split_name}.jsonl", split_records_list)
        write_csv(args.output_dir / f"{split_name}.csv", split_records_list)

    summary = {
        "input": {
            "p_folio_csv": str(args.p_folio_csv),
            "folio_csv": str(args.folio_csv) if args.folio_csv.exists() else None,
            "folio_jsonl": [str(path) for path in folio_jsonl_paths],
        },
        "options": {
            "seed": args.seed,
            "split_ratios": args.split_ratios,
            "split_by": "sample" if args.split_by_sample else "story_id",
            "drop_empty_proofs": args.drop_empty_proofs,
            "use_corrected_derivations": not args.no_corrected,
            "coerce_commented_labels": args.coerce_commented_labels,
        },
        "parse_stats": parse_stats,
        "folio_stats": folio_stats,
        "build_stats": build_stats,
        "split_counts": {split: len(items) for split, items in sorted(records_by_split.items())},
        "outputs": {
            "all_data_jsonl": str(args.output_dir / "all_data.jsonl"),
            "all_data_csv": str(args.output_dir / "all_data.csv"),
            "train_jsonl": str(args.output_dir / "train.jsonl"),
            "train_csv": str(args.output_dir / "train.csv"),
            "validation_jsonl": str(args.output_dir / "validation.jsonl"),
            "validation_csv": str(args.output_dir / "validation.csv"),
            "test_jsonl": str(args.output_dir / "test.jsonl"),
            "test_csv": str(args.output_dir / "test.csv"),
        },
    }

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
