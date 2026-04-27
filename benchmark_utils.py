from unsloth import FastVisionModel
from transformers import TrainerCallback
import ast
import pandas as pd
import torch
import csv
import os
import re
from pathlib import Path

def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def _extract_mir_answer(text):
    cleaned = str(text).replace("```python", "").replace("```", "").strip()
    output_markers = list(re.finditer(r'output\s*:', cleaned, flags=re.IGNORECASE))
    if output_markers:
        cleaned = cleaned[output_markers[-1].end():].strip()
    return cleaned


def _normalize_mir_ground_truth(value):
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value).strip()


def _mir_answers_match(ground_truth, prediction):
    answer = _normalize_mir_ground_truth(ground_truth)
    verdict = _extract_mir_answer(prediction)

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

def evaluate_binary_answer(model, tokenizer, bench_folder, results_folder, step, csv_name, dataset_type, trainer=None):
    """Evaluation for binary (yes/no) questions."""
    print(f"  [Binary] Evaluating {dataset_type} set...")
    
    results_file = os.path.join(results_folder, f"results_bench_binary_{dataset_type}_{step}.csv")
    bench_data_file = os.path.join(bench_folder, csv_name)
    
    prompt_path = os.path.join(os.path.dirname(bench_folder), "prompt_test.txt")

    content = load_text_file(prompt_path)
    
    try:
        with open(bench_data_file, 'r', newline='') as outputs, \
             open(results_file, 'w', newline='') as results:
            reader = csv.reader(outputs)
            next(reader, None)
            writer = csv.writer(results)
            writer.writerow(["question", "label", "answer"])
            
            for n, row in enumerate(reader):
                if len(row) < 3: continue
                messages = [{"role": "user", "content": content.format(question=row[1])}]
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                model_inputs = tokenizer(text=[text], return_tensors="pt", add_special_tokens=False).to(model.device)

                generated_ids = model.generate(
                    **model_inputs,
                    do_sample=False, temperature=0, 
                    max_new_tokens=8,
                )

                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                writer.writerow([row[1], row[2], generated_text])

                del model_inputs, generated_ids, output_ids, text
                torch.cuda.empty_cache()

        results_df = pd.read_csv(results_file)
        TP, FP, TN, FN, UNKNOWN = 0, 0, 0, 0, 0

        for _, row in results_df.iterrows():
            y_true = str(row['label']).strip()
            y_pred = str(row['answer']).lower()
            pred_yes = re.search(r'\byes\b', y_pred) is not None
            pred_no = re.search(r'\bno\b', y_pred) is not None

            if y_true == "yes" and pred_yes and not pred_no: TP += 1
            elif y_true == "no" and pred_no and not pred_yes: TN += 1
            elif y_true == "no" and pred_yes and not pred_no: FP += 1
            elif y_true == "yes" and pred_no and not pred_yes: FN += 1
            else: UNKNOWN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
        print(f"  Binary Accuracy ({dataset_type}): {accuracy:.2f}% (TP:{TP} TN:{TN} FP:{FP} FN:{FN} U:{UNKNOWN})")

        if trainer:
            trainer.log({
                f"bench_binary_{dataset_type}/accuracy": accuracy,
                f"bench_binary_{dataset_type}/tp": TP,
                f"bench_binary_{dataset_type}/tn": TN,
                f"bench_binary_{dataset_type}/fp": FP,
                f"bench_binary_{dataset_type}/fn": FN,
                f"bench_binary_{dataset_type}/unknowns": UNKNOWN,
            })

        # Save aggregate metrics
        metrics_file = os.path.join(results_folder, f"metrics_binary_{dataset_type}_summary.csv")
        file_exists = os.path.exists(metrics_file)
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "TP", "FP", "FN", "TN", "UNKNOWN", "total", "accuracy"])
            writer.writerow([step, TP, FP, FN, TN, UNKNOWN, len(results_df), f"{accuracy:.2f}"])

    except Exception as e:
        print(f"Error in evaluate_binary_answer: {e}")

def evaluate_multiple_choice(model, tokenizer, bench_folder, results_folder, step, csv_name, dataset_type, trainer=None):
    """Evaluation for multiple choice (a/b/c) questions."""
    print(f"  [MC] Evaluating {dataset_type} set...")
    
    results_file = os.path.join(results_folder, f"results_bench_mc_{dataset_type}_{step}.csv")
    bench_data_file = os.path.join(bench_folder, csv_name)
    
    prompt_path = os.path.join(os.path.dirname(bench_folder), "prompt_test.txt")

    content = load_text_file(prompt_path)
    
    try:
        with open(bench_data_file, 'r', newline='') as outputs, \
             open(results_file, 'w', newline='') as results:
            reader = csv.reader(outputs)
            next(reader, None)
            writer = csv.writer(results)
            writer.writerow(["question", "label", "answer"])
            
            for n, row in enumerate(reader):
                if len(row) < 3: continue
                messages = [{"role": "user", "content": content.format(question=row[1])}]
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                model_inputs = tokenizer(text=[text], return_tensors="pt", add_special_tokens=False).to(model.device)

                generated_ids = model.generate(
                    **model_inputs,
                    do_sample=False, temperature=0, 
                    max_new_tokens=8,
                )

                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                writer.writerow([row[1], row[2], generated_text])

                del model_inputs, generated_ids, output_ids, text
                torch.cuda.empty_cache()

        results_df = pd.read_csv(results_file)
        correct, total, unknown = 0, 0, 0
        labels = ['a', 'b', 'c']
        matrix = {t: {p: 0 for p in labels} for t in labels}

        for _, row in results_df.iterrows():
            y_true = str(row['label']).strip().lower()
            y_pred = str(row['answer']).strip().lower()
            match = re.search(r'\b([abc])\b', y_pred)
            if match:
                pred_label = match.group(1)
                if pred_label == y_true: correct += 1
                if y_true in labels: matrix[y_true][pred_label] += 1
                total += 1
            else:
                unknown += 1

        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"  MC Accuracy ({dataset_type}): {accuracy:.2f}% (C:{correct} I:{total-correct} U:{unknown})")

        if trainer:
            trainer.log({
                f"bench_mc_{dataset_type}/accuracy": accuracy,
                f"bench_mc_{dataset_type}/unknowns": unknown,
            })

        # Save aggregate metrics
        metrics_file = os.path.join(results_folder, f"metrics_mc_{dataset_type}_summary.csv")
        file_exists = os.path.exists(metrics_file)
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "correct", "incorrect", "unknown", "total", "accuracy"])
            writer.writerow([step, correct, total - correct, unknown, len(results_df), f"{accuracy:.2f}"])

    except Exception as e:
        print(f"Error in evaluate_multiple_choice: {e}")


def evaluate_mir_benchmark(model, tokenizer, bench_file, results_folder, step, trainer=None):
    """Evaluation for MIR-Bench parquet files."""
    try:
        dataset_name = Path(bench_file).stem
        print(f"  [MIR] Evaluating {dataset_name}...")

        results_file = os.path.join(results_folder, f"results_mir_{dataset_name}_{step}.csv")
        bench_df = pd.read_parquet(bench_file)

        required_columns = {"prompt", "answer"}
        missing_columns = required_columns - set(bench_df.columns)
        if missing_columns:
            raise ValueError(f"MIR-Bench file {bench_file} is missing required columns: {sorted(missing_columns)}")

        with open(results_file, 'w', newline='') as results:
            writer = csv.writer(results)
            writer.writerow(["num_shots", "answer", "prediction", "extracted_answer", "correct"])

            total_correct = 0
            total_count = 0
            by_shots = {}

            for _, row in bench_df.iterrows():
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
                    max_new_tokens=256,
                )

                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

                is_correct, extracted_answer = _mir_answers_match(row["answer"], generated_text)
                num_shots = int(row["num_shots"]) if "num_shots" in row and pd.notna(row["num_shots"]) else max(prompt.count("Input:") - 1, 0)

                writer.writerow([
                    num_shots,
                    _normalize_mir_ground_truth(row["answer"]),
                    generated_text,
                    extracted_answer,
                    int(is_correct),
                ])

                stats = by_shots.setdefault(num_shots, {"correct": 0, "total": 0})
                stats["correct"] += int(is_correct)
                stats["total"] += 1
                total_correct += int(is_correct)
                total_count += 1

                del model_inputs, generated_ids, output_ids, text
                torch.cuda.empty_cache()

        accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0
        print(f"  MIR Accuracy ({dataset_name}): {accuracy:.2f}% ({total_correct}/{total_count})")

        if trainer:
            log_payload = {f"bench_mir_{dataset_name}/accuracy": accuracy}
            for shot_count, stats in sorted(by_shots.items()):
                shot_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                log_payload[f"bench_mir_{dataset_name}/{shot_count}_shot_accuracy"] = shot_accuracy
            trainer.log(log_payload)

        metrics_file = os.path.join(results_folder, f"metrics_mir_{dataset_name}_summary.csv")
        file_exists = os.path.exists(metrics_file)
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "correct", "total", "accuracy"])
            writer.writerow([step, total_correct, total_count, f"{accuracy:.2f}"])

        shot_metrics_file = os.path.join(results_folder, f"metrics_mir_{dataset_name}_by_shot_summary.csv")
        shot_file_exists = os.path.exists(shot_metrics_file)
        with open(shot_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not shot_file_exists:
                writer.writerow(["step", "num_shots", "correct", "total", "accuracy"])
            for shot_count, stats in sorted(by_shots.items()):
                shot_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                writer.writerow([step, shot_count, stats["correct"], stats["total"], f"{shot_accuracy:.2f}"])

    except Exception as e:
        print(f"Error in evaluate_mir_benchmark ({bench_file}): {e}")


def run_benchmark_evaluation(model, tokenizer, ba_bench_folder, mc_bench_folder, results_folder_name, step, trainer=None, mir_bench_path=None):
    """Unified evaluation runner for binary, multiple choice, and MIR benchmarks."""
    print(f"\n[Benchmark] Step {step} evaluation...")
    FastVisionModel.for_inference(model)
    
    with torch.inference_mode():
        for csv_name in ["bench_train.csv", "bench_val.csv"]:
            dataset_type = "train" if "train" in csv_name else "val"
            
            # Binary Answer Benchmark
            ba_results_folder = os.path.join(ba_bench_folder, f"results_{results_folder_name}")
            os.makedirs(ba_results_folder, exist_ok=True)
            evaluate_binary_answer(model, tokenizer, ba_bench_folder, ba_results_folder, step, csv_name, dataset_type, trainer)
            
            # Multiple Choice Benchmark
            mc_results_folder = os.path.join(mc_bench_folder, f"results_{results_folder_name}")
            os.makedirs(mc_results_folder, exist_ok=True)
            evaluate_multiple_choice(model, tokenizer, mc_bench_folder, mc_results_folder, step, csv_name, dataset_type, trainer)

        if mir_bench_path:
            mir_path = Path(mir_bench_path)
            mir_results_folder = os.path.join(str(mir_path.parent if mir_path.is_file() else mir_path), f"results_{results_folder_name}")
            os.makedirs(mir_results_folder, exist_ok=True)

            mir_files = [mir_path] if mir_path.is_file() else sorted(mir_path.glob("*.parquet"))
            if not mir_files:
                print(f"  [MIR] No parquet files found in {mir_bench_path}, skipping.")
            for mir_file in mir_files:
                evaluate_mir_benchmark(model, tokenizer, str(mir_file), mir_results_folder, step, trainer)

class BenchmarkCallback(TrainerCallback):
    def __init__(self, ba_bench_folder, mc_bench_folder, tokenizer, lora_folder, eval_steps=100, trainer=None, mir_bench_path=None):
        self.ba_bench_folder = ba_bench_folder
        self.mc_bench_folder = mc_bench_folder
        self.tokenizer = tokenizer
        self.lora_folder = lora_folder
        self.eval_steps = eval_steps
        self.trainer = trainer
        self.mir_bench_path = mir_bench_path

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            model = kwargs['model']
            run_benchmark_evaluation(
                model=model,
                tokenizer=self.tokenizer,
                ba_bench_folder=self.ba_bench_folder,
                mc_bench_folder=self.mc_bench_folder,
                results_folder_name=self.lora_folder,
                step=state.global_step,
                trainer=self.trainer,
                mir_bench_path=self.mir_bench_path,
            )
            FastVisionModel.for_training(model)
