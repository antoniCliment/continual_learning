import argparse
import csv
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


# This benchmark runs one short generation per next-step item. Avoid Unsloth /
# TorchInductor compile overhead unless the caller explicitly opts in.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")


MODEL_PATH = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
BENCHMARK_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = BENCHMARK_DIR / "data" / "processed"
RESULTS_DIR = BENCHMARK_DIR / "results"
BENCH_FILE = PROCESSED_DATA_DIR / "test_next_step.jsonl"
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 256

NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?%?"
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_records(path, limit=None):
    records = []
    with open(path, "r", encoding="utf-8") as bench:
        for line_no, line in enumerate(bench, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            required_keys = {
                "id",
                "question",
                "answer",
                "evaluation",
                "problem_index",
                "step_index",
                "total_steps",
                "next_step_question",
            }
            missing_keys = required_keys - set(record)
            if missing_keys:
                raise ValueError(
                    f"{path}:{line_no} is missing keys: {sorted(missing_keys)}"
                )
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def extract_answer_text(text):
    cleaned = str(text).replace("```python", "").replace("```", "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()

    marker_pattern = re.compile(
        r"(?:final answer|answer|output)\s*:\s*",
        flags=re.IGNORECASE,
    )
    markers = list(marker_pattern.finditer(cleaned))
    if markers:
        cleaned = cleaned[markers[-1].end() :].strip()

    return cleaned


def normalize_number_text(value):
    return str(value).strip().replace("$", "").replace(",", "")


def extract_numbers(text):
    return [normalize_number_text(match.group(0)) for match in NUMBER_RE.finditer(text)]


def to_decimal(value):
    text = normalize_number_text(value)
    if text.endswith("%"):
        text = text[:-1]
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return Decimal(numerator) / Decimal(denominator)
    return Decimal(text)


def numeric_equal(expected, predicted):
    try:
        expected_num = to_decimal(expected)
        predicted_num = to_decimal(predicted)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return normalize_number_text(expected) == normalize_number_text(predicted)

    tolerance = max(abs(expected_num), Decimal("1")) * Decimal("1e-9")
    return abs(expected_num - predicted_num) <= tolerance


def normalize_text(text):
    text = extract_answer_text(text).lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(TOKEN_RE.findall(text))


def token_f1(expected, predicted):
    expected_tokens = TOKEN_RE.findall(normalize_text(expected))
    predicted_tokens = TOKEN_RE.findall(normalize_text(predicted))
    if not expected_tokens or not predicted_tokens:
        return 0.0

    expected_counts = {}
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1

    overlap = 0
    for token in predicted_tokens:
        count = expected_counts.get(token, 0)
        if count:
            overlap += 1
            expected_counts[token] = count - 1

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_prediction(record, prediction):
    extracted = extract_answer_text(prediction)
    evaluation = record["evaluation"]
    expected_target = str(evaluation.get("target", "")).strip()
    kind = evaluation.get("kind", "text")

    if kind == "numeric":
        numbers = extract_numbers(extracted)
        extracted_target = numbers[-1] if numbers else ""
        correct = bool(extracted_target) and numeric_equal(expected_target, extracted_target)
        return correct, extracted, extracted_target

    expected_text = str(record["answer"])
    expected_norm = normalize_text(expected_text)
    predicted_norm = normalize_text(extracted)
    f1 = token_f1(expected_text, extracted)
    correct = (
        bool(expected_norm)
        and (expected_norm in predicted_norm or predicted_norm in expected_norm or f1 >= 0.75)
    )
    return correct, extracted, f"text_f1={f1:.3f}"


def update_stats(stats, key, correct):
    bucket = stats.setdefault(key, {"correct": 0, "total": 0})
    bucket["correct"] += int(correct)
    bucket["total"] += 1


def write_metric_rows(writer, group_name, stats):
    for key, values in sorted(stats.items(), key=lambda item: str(item[0])):
        total = values["total"]
        correct = values["correct"]
        accuracy = (correct / total) * 100 if total else 0
        writer.writerow([group_name, key, correct, total, f"{accuracy:.2f}"])


def apply_chat_template(tokenizer, prompt, thinking):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def clear_accelerator_cache(torch_module):
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    if hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
        torch_module.xpu.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma 4 on the GSM8K next-step benchmark."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--bench-file", type=Path, default=BENCH_FILE)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode when the tokenizer chat template supports it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.bench_file.exists():
        print(f"Benchmark file not found: {args.bench_file}")
        print("Run: python3 scripts/build_next_step_benchmark.py")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.bench_file, limit=args.limit)
    if not records:
        print(f"No records found in {args.bench_file}")
        sys.exit(1)

    import torch
    from unsloth import FastLanguageModel

    print(f"Loading model: {args.model_path}")
    print(f"Benchmark file: {args.bench_file}")
    print(f"Thinking mode: {'enabled' if args.thinking else 'disabled'}")
    print(f"Max new tokens: {args.max_new_tokens}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    FastLanguageModel.for_inference(model)

    model_name = os.path.basename(os.path.normpath(args.model_path))
    thinking_suffix = "_thinking" if args.thinking else ""
    results_file = args.output_dir / (
        f"results_{args.bench_file.stem}_{model_name}{thinking_suffix}.csv"
    )
    metrics_file = args.output_dir / (
        f"metrics_{args.bench_file.stem}_{model_name}{thinking_suffix}.csv"
    )

    print(f"Evaluating {len(records)} next-step prompts...")

    all_stats = {"all": {"correct": 0, "total": 0}}
    by_source = {}
    by_step = {}
    by_final_step = {}
    unknown_numeric = 0

    with torch.inference_mode(), open(
        results_file, "w", newline="", encoding="utf-8"
    ) as results:
        writer = csv.writer(results)
        writer.writerow(
            [
                "id",
                "problem_index",
                "step_index",
                "total_steps",
                "is_final_step",
                "next_step_question",
                "expected_answer",
                "evaluation_kind",
                "evaluation_source",
                "evaluation_target",
                "prediction",
                "extracted_answer",
                "extracted_target",
                "correct",
            ]
        )

        for idx, record in enumerate(records, start=1):
            prompt = record["question"]
            text = apply_chat_template(tokenizer, prompt, args.thinking)
            model_inputs = tokenizer(
                text=[text],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
            generated_text = tokenizer.decode(
                output_ids,
                skip_special_tokens=not args.thinking,
            ).strip()

            correct, extracted, extracted_target = score_prediction(record, generated_text)
            evaluation = record["evaluation"]
            source = evaluation.get("source", "unknown")

            update_stats(all_stats, "all", correct)
            update_stats(by_source, source, correct)
            update_stats(by_step, record["step_index"], correct)
            update_stats(by_final_step, "final" if record["is_final_step"] else "intermediate", correct)
            if evaluation.get("kind") == "numeric" and not extracted_target:
                unknown_numeric += 1

            writer.writerow(
                [
                    record["id"],
                    record["problem_index"],
                    record["step_index"],
                    record["total_steps"],
                    int(record["is_final_step"]),
                    record["next_step_question"],
                    record["answer"],
                    evaluation.get("kind", ""),
                    source,
                    evaluation.get("target", ""),
                    generated_text,
                    extracted,
                    extracted_target,
                    int(correct),
                ]
            )

            print(f"Processed {idx}/{len(records)}", end="\r")

            del model_inputs, generated_ids, output_ids, text
            clear_accelerator_cache(torch)

    print()

    with open(metrics_file, "w", newline="", encoding="utf-8") as metrics:
        writer = csv.writer(metrics)
        writer.writerow(["group", "key", "correct", "total", "accuracy"])
        write_metric_rows(writer, "overall", all_stats)
        write_metric_rows(writer, "evaluation_source", by_source)
        write_metric_rows(writer, "step_index", by_step)
        write_metric_rows(writer, "step_type", by_final_step)
        writer.writerow(["unknown", "numeric_no_extraction", unknown_numeric, len(records), ""])

    overall = all_stats["all"]
    overall_accuracy = (
        (overall["correct"] / overall["total"]) * 100 if overall["total"] else 0
    )
    print(f"Results file: {results_file}")
    print(f"Metrics file: {metrics_file}")
    print(
        f"Overall next-step accuracy: {overall_accuracy:.2f}% "
        f"({overall['correct']}/{overall['total']})"
    )


if __name__ == "__main__":
    main()
