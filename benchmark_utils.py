from unsloth import FastVisionModel
from transformers import TrainerCallback
import pandas as pd
import torch
import csv
import os
import re

def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

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

def run_benchmark_evaluation(model, tokenizer, ba_bench_folder, mc_bench_folder, results_folder_name, step, trainer=None):
    """Unified evaluation runner for both binary and multiple choice."""
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

class BenchmarkCallback(TrainerCallback):
    def __init__(self, ba_bench_folder, mc_bench_folder, tokenizer, lora_folder, eval_steps=100, trainer=None):
        self.ba_bench_folder = ba_bench_folder
        self.mc_bench_folder = mc_bench_folder
        self.tokenizer = tokenizer
        self.lora_folder = lora_folder
        self.eval_steps = eval_steps
        self.trainer = trainer

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
                trainer=self.trainer
            )
            FastVisionModel.for_training(model)
