from unsloth import FastLanguageModel
import torch
import sys
import csv
import pandas as pd
import os
import re
import traceback

# ------------------------------------------------------------------------
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True   # Use 4bit quantization to reduce memory usage. Can be False.

def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def evaluate_file(model, tokenizer, bench_data_file, results_file, prompt_template):
    if not os.path.exists(bench_data_file):
        print(f"Warning: Benchmark data file not found at {bench_data_file}")
        return

    print(f"\nStarting evaluation on {bench_data_file}...")
    
    with torch.inference_mode():
        try:
            with open(bench_data_file, 'r', newline='') as outputs, \
                 open(results_file, 'w', newline='') as results:
                reader = csv.reader(outputs)
                header = next(reader, None) # Skip header
                writer = csv.writer(results)
                writer.writerow(["question", "label", "answer"])
                
                for n, row in enumerate(reader):
                    if len(row) < 3: continue
                    messages = [
                        {"role": "user", "content": prompt_template.format(question=row[1])},
                    ]

                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    model_inputs = tokenizer(text=[text], return_tensors="pt", add_special_tokens=False).to(model.device)

                    generated_ids = model.generate(
                        **model_inputs,
                        do_sample=False, 
                        temperature=0, 
                        max_new_tokens=10
                    )

                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    writer.writerow([row[1], row[2], generated_text])
                    print(f"Processed row {n}", end="\r")

                    # Explicitly free memory
                    del model_inputs, generated_ids, output_ids, text
                    torch.cuda.empty_cache()

            print("\nEvaluation complete. Calculating metrics...")
            
            # Calculate Accuracy and Confusion Matrix
            results_df = pd.read_csv(results_file)
            correct = 0
            total = 0
            unknown = 0
            
            # Labels for the matrix
            labels = ['a', 'b', 'c']
            # Matrix: [True][Pred]
            matrix = {t: {p: 0 for p in labels} for t in labels}

            for _, row in results_df.iterrows():
                y_true = str(row['label']).strip().lower()
                y_pred = str(row['answer']).strip().lower()

                # Robust extraction of a, b, or c
                match = re.search(r'\b([abc])\b', y_pred)
                if match:
                    pred_label = match.group(1)
                    if pred_label == y_true:
                        correct += 1
                    
                    if y_true in labels:
                        matrix[y_true][pred_label] += 1
                    total += 1
                else:
                    unknown += 1

            # Use raw string or double escape for backslash to avoid SyntaxWarning
            print(f"\nResults Matrix (True \\ Pred) for {os.path.basename(bench_data_file)}:")
            header = "      " + "     ".join(labels)
            print(header)
            for t in labels:
                row_str = f" {t}: "
                for p in labels:
                    row_str += f" {matrix[t][p]:5d} "
                print(row_str)

            print(f"\nStatistics for {os.path.basename(bench_data_file)}:")
            print(f"Correct: {correct}")
            print(f"Incorrect: {total - correct}")
            print(f"Unknown: {unknown}/{len(results_df)}")
            accuracy = (correct / total) * 100 if total > 0 else 0
            print(f"Accuracy: {accuracy:.2f}%")

        except Exception as e:
            print(f"Error during benchmark evaluation of {bench_data_file}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_bench.py [model_name/lora_folder] [unused_lora_folder] [benchmark_folder]")
        print("Note: Unsloth can load either the base model or the lora folder directly as model_name.")
        print("Example: python test_bench.py ../../models/gemma3-4b-rhinolume_v3 dummy gen_v1")
        sys.exit(1)
    
    # In unsloth, if we want to load a lora model, we just pass the lora folder as model_name
    model_id, lora_folder, bench_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # If the lora_folder exists and contains an adapter config, load from it.
    # Otherwise, load the base model_id.
    if os.path.isdir(lora_folder) and os.path.exists(os.path.join(lora_folder, "adapter_config.json")):
        load_path = lora_folder
        print(f"Loading fine-tuned model (LoRA) from: {load_path}")
    else:
        load_path = model_id
        print(f"LoRA folder invalid or not found, loading base model: {load_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = load_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Set model to inference mode
    FastLanguageModel.for_inference(model)

    prompt_path = os.path.join("./benchmarks/rhinolume/multiple_choice/prompt_test.txt")
    prompt_template = load_text_file(prompt_path)

    # Evaluate both train and val benchmarks
    for split in ["train", "val"]:
        bench_data_file = os.path.join(bench_folder, f"bench_{split}.csv")
        results_file = os.path.join(bench_folder, f"results_bench_{split}.csv")
        evaluate_file(model, tokenizer, bench_data_file, results_file, prompt_template)
