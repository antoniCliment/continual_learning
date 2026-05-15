import ast
import csv
import os
import re
import sys

# This benchmark runs one prompt at a time, so avoid Unsloth/TorchInductor's
# compile path unless the caller explicitly enables it in the environment.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import pandas as pd
import torch


max_seq_length = 4096
dtype = None
load_in_4bit = True


def extract_answer(text):
    cleaned = str(text).replace("```python", "").replace("```", "").strip()
    output_markers = list(re.finditer(r"output\s*:", cleaned, flags=re.IGNORECASE))
    if output_markers:
        cleaned = cleaned[output_markers[-1].end():].strip()
    return cleaned


def normalize_ground_truth(value):
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value).strip()


def answers_match(ground_truth, prediction):
    answer = normalize_ground_truth(ground_truth)
    verdict = extract_answer(prediction)

    if verdict == answer:
        return True, verdict

    try:
        parsed = ast.literal_eval(verdict)
        if isinstance(parsed, dict) and len(parsed) == 1:
            parsed = parsed[next(iter(parsed))]
        if isinstance(parsed, set) and len(parsed) == 1:
            parsed = next(iter(parsed))
        if str(parsed) == answer:
            return True, verdict
    except Exception:
        pass

    return False, verdict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_bench.py [model_name_or_lora_path] [parquet_file]")
        print("Example 2: python test_bench.py unsloth/gemma-4-E4B-it-unsloth-bnb-4bit ./benchmarks/mir/MIR-Core.parquet")
        sys.exit(1)

    model_path = sys.argv[1]
    bench_file = sys.argv[2] if len(sys.argv) > 2 else "./benchmarks/mir/MIR-Core.parquet"

    if not os.path.exists(bench_file):
        print(f"Benchmark file not found: {bench_file}")
        sys.exit(1)

    from unsloth import FastLanguageModel

    print(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    FastLanguageModel.for_inference(model)

    bench_df = pd.read_parquet(bench_file)
    required_columns = {"prompt", "answer"}
    missing_columns = required_columns - set(bench_df.columns)
    if missing_columns:
        print(f"Missing required columns in {bench_file}: {sorted(missing_columns)}")
        sys.exit(1)

    dataset_name = os.path.splitext(os.path.basename(bench_file))[0]
    model_name = os.path.basename(os.path.normpath(model_path))
    results_file = os.path.join(os.path.dirname(bench_file), f"results_{dataset_name}_{model_name}.csv")
    metrics_file = os.path.join(os.path.dirname(bench_file), f"metrics_{dataset_name}_{model_name}.csv")

    print(f"Evaluating {dataset_name} on {len(bench_df)} prompts...")

    total_correct = 0
    by_shots = {}
    total_count = 0

    with torch.inference_mode(), open(results_file, "w", newline="") as results:
        writer = csv.writer(results)
        writer.writerow(["num_shots", "answer", "prediction", "extracted_answer", "correct"])

        for idx, row in bench_df.iterrows():
            if total_count >= 3:
                break

            prompt = str(row["prompt"])
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = tokenizer(text=[text], return_tensors="pt", add_special_tokens=False).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                temperature=0,
                max_new_tokens=8092,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            is_correct, extracted = answers_match(row["answer"], generated_text)
            num_shots = int(row["num_shots"]) if "num_shots" in row and pd.notna(row["num_shots"]) else max(prompt.count("Input:") - 1, 0)

            writer.writerow([
                num_shots,
                normalize_ground_truth(row["answer"]),
                generated_text,
                extracted,
                int(is_correct),
            ])

            stats = by_shots.setdefault(num_shots, {"correct": 0, "total": 0})
            stats["correct"] += int(is_correct)
            stats["total"] += 1
            total_correct += int(is_correct)
            total_count += 1

            print(f"Processed {total_count}/{len(bench_df)}", end="\r")

            del model_inputs, generated_ids, output_ids, text
            torch.cuda.empty_cache()

    print()

    accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0

    with open(metrics_file, "w", newline="") as metrics:
        writer = csv.writer(metrics)
        writer.writerow(["num_shots", "correct", "total", "accuracy"])
        for num_shots, stats in sorted(by_shots.items()):
            shot_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            writer.writerow([num_shots, stats["correct"], stats["total"], f"{shot_accuracy:.2f}"])
        writer.writerow(["ALL", total_correct, total_count, f"{accuracy:.2f}"])

    print(f"Results file: {results_file}")
    print(f"Metrics file: {metrics_file}")
    print(f"Overall accuracy: {accuracy:.2f}% ({total_correct}/{total_count})")
    for num_shots, stats in sorted(by_shots.items()):
        shot_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{num_shots}-shot accuracy: {shot_accuracy:.2f}% ({stats['correct']}/{stats['total']})")
