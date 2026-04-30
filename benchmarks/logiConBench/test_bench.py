import csv
import json
import os
import random
import re
import sys
from pathlib import Path

# This benchmark runs short, single-prompt generations. Avoid Unsloth/TorchInductor
# compile overhead unless the caller explicitly enables it.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch


MODEL_PATH = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
BENCH_FILE = Path(__file__).with_name("2statements.jsonl")

MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 16
RANDOM_SEED = 3407

# The LogiConBench paper reports Task 1 with 500 consistent and 500 inconsistent
# label-list prompts per configuration.
SAMPLES_PER_CLASS = 500


def safe_model_name(model_path):
    return os.path.basename(os.path.normpath(model_path)).replace("/", "-")


def torch_accelerator_available():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return True
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def require_unsloth_accelerator():
    if torch_accelerator_available():
        return

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        "Unsloth requires a visible NVIDIA, AMD ROCm, or Intel GPU, but this "
        "Python process cannot see one.",
        file=sys.stderr,
    )
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", file=sys.stderr)
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}", file=sys.stderr)
    print("Run this benchmark in a GPU-enabled shell/container.", file=sys.stderr)
    sys.exit(1)


def reject_gguf_model(model_path):
    if model_path.lower().endswith("-gguf") or model_path.lower().endswith(".gguf"):
        print(
            "This benchmark loads PyTorch/Transformers checkpoints through Unsloth, "
            "but the requested model is a GGUF checkpoint/repository."
        )
        print("Set MODEL_PATH to a non-GGUF checkpoint before running this script.")
        sys.exit(1)


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as bench:
        for line_no, line in enumerate(bench, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            required_keys = {"nl_nodes", "valid_sets", "invalid_sets"}
            missing_keys = required_keys - set(record)
            if missing_keys:
                raise ValueError(
                    f"{path}:{line_no} is missing keys: {sorted(missing_keys)}"
                )
            records.append((line_no, record))
    return records


def normalize_assignment(assignment):
    return [str(label).strip().upper() for label in assignment]


def build_eval_items(records):
    consistent = []
    inconsistent = []

    for line_no, record in records:
        nl_nodes = record["nl_nodes"]
        for assignment in record["valid_sets"]:
            consistent.append((line_no, nl_nodes, normalize_assignment(assignment), "yes"))
        for assignment in record["invalid_sets"]:
            inconsistent.append((line_no, nl_nodes, normalize_assignment(assignment), "no"))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(consistent)
    rng.shuffle(inconsistent)

    if SAMPLES_PER_CLASS is not None:
        consistent = consistent[:SAMPLES_PER_CLASS]
        inconsistent = inconsistent[:SAMPLES_PER_CLASS]

    eval_items = consistent + inconsistent
    rng.shuffle(eval_items)
    return eval_items


def format_assignment(assignment):
    return "[" + ", ".join(assignment) + "]"


def build_prompt(statements, assignment):
    statement_lines = "\n".join(
        f"{idx}. {statement}" for idx, statement in enumerate(statements, start=1)
    )
    label_lines = "\n".join(
        f"{idx}. {label}" for idx, label in enumerate(assignment, start=1)
    )
    return (
        "You are evaluating logical consistency.\n"
        "Each statement has one Boolean label: T means the statement is true, "
        "and F means the statement is false.\n\n"
        "Statements:\n"
        f"{statement_lines}\n\n"
        "Boolean labels:\n"
        f"{label_lines}\n\n"
        "Can all of these labels hold at the same time without contradiction?\n"
        'Answer only with exactly one word: "yes" or "no". Nothing else.'
    )


def extract_yes_no(text):
    cleaned = str(text).replace("```", "").strip()
    output_markers = list(
        re.finditer(r"(?:answer|output)\s*:\s*", cleaned, flags=re.IGNORECASE)
    )
    if output_markers:
        cleaned = cleaned[output_markers[-1].end():].strip()

    match = re.search(r"\b(yes|no)\b", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


if __name__ == "__main__":
    reject_gguf_model(MODEL_PATH)

    if not BENCH_FILE.exists():
        print(f"Benchmark file not found: {BENCH_FILE}")
        sys.exit(1)

    require_unsloth_accelerator()

    from unsloth import FastLanguageModel

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    FastLanguageModel.for_inference(model)

    records = load_records(BENCH_FILE)
    eval_items = build_eval_items(records)

    dataset_name = BENCH_FILE.stem
    model_name = safe_model_name(MODEL_PATH)
    results_file = BENCH_FILE.with_name(f"results_task1_{dataset_name}_{model_name}.csv")
    metrics_file = BENCH_FILE.with_name(f"metrics_task1_{dataset_name}_{model_name}.csv")

    print(
        f"Evaluating LogiConBench Task 1 on {len(eval_items)} prompts "
        f"from {BENCH_FILE.name}..."
    )

    totals = {
        "consistent": {"correct": 0, "total": 0},
        "inconsistent": {"correct": 0, "total": 0},
        "all": {"correct": 0, "total": 0},
    }
    unknown = 0

    with torch.inference_mode(), open(
        results_file, "w", newline="", encoding="utf-8"
    ) as results:
        writer = csv.writer(results)
        writer.writerow(
            [
                "line_no",
                "statements",
                "assignment",
                "expected",
                "prediction",
                "extracted_answer",
                "correct",
            ]
        )

        for idx, (line_no, statements, assignment, expected) in enumerate(eval_items, start=1):
            prompt = build_prompt(statements, assignment)
            messages = [{"role": "user", "content": prompt}]
            text = apply_chat_template(tokenizer, messages)
            model_inputs = tokenizer(
                text=[text],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                temperature=0,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            extracted = extract_yes_no(generated_text)
            is_correct = extracted == expected

            if not extracted:
                unknown += 1

            split = "consistent" if expected == "yes" else "inconsistent"
            totals[split]["correct"] += int(is_correct)
            totals[split]["total"] += 1
            totals["all"]["correct"] += int(is_correct)
            totals["all"]["total"] += 1

            writer.writerow(
                [
                    line_no,
                    json.dumps(statements, ensure_ascii=False),
                    format_assignment(assignment),
                    expected,
                    generated_text,
                    extracted,
                    int(is_correct),
                ]
            )

            print(f"Processed {idx}/{len(eval_items)}", end="\r")

            del model_inputs, generated_ids, output_ids, text
            torch.cuda.empty_cache()

    print()

    with open(metrics_file, "w", newline="", encoding="utf-8") as metrics:
        writer = csv.writer(metrics)
        writer.writerow(["split", "correct", "total", "accuracy"])
        for split in ["consistent", "inconsistent", "all"]:
            correct = totals[split]["correct"]
            total = totals[split]["total"]
            accuracy = (correct / total) * 100 if total else 0
            writer.writerow([split, correct, total, f"{accuracy:.2f}"])
        writer.writerow(["unknown", unknown, totals["all"]["total"], ""])

    overall = totals["all"]
    overall_accuracy = (
        (overall["correct"] / overall["total"]) * 100 if overall["total"] else 0
    )
    print(f"Results file: {results_file}")
    print(f"Metrics file: {metrics_file}")
    print(
        f"Overall accuracy: {overall_accuracy:.2f}% "
        f"({overall['correct']}/{overall['total']})"
    )
    for split in ["consistent", "inconsistent"]:
        correct = totals[split]["correct"]
        total = totals[split]["total"]
        accuracy = (correct / total) * 100 if total else 0
        print(f"{split.title()} accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Unknown outputs: {unknown}")
