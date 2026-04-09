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

    prompt_path = os.path.join("./benchmarks/rhinolume/binary_answer/prompt_test.txt")

    content = load_text_file(prompt_path)
    results_file = os.path.join(bench_folder, "results_bench.csv")
    bench_data_file = os.path.join(bench_folder, "bench_train.csv")

    if not os.path.exists(bench_data_file):
        print(f"Warning: Benchmark data file not found at {bench_data_file}")
        sys.exit(1)

    print(f"Starting evaluation on {bench_data_file}...")
    
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
                        {"role": "user", "content": content.format(question=row[1])},
                    ]

                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    model_inputs = tokenizer(text=[text], return_tensors="pt", add_special_tokens=False).to(model.device)

                    generated_ids = model.generate(
                        **model_inputs,
                        do_sample=False, 
                        temperature=0, 
                        max_new_tokens=128
                    )

                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    writer.writerow([row[1], row[2], generated_text])
                    print(f"Processed row {n}", end="\r")

                    # Explicitly free memory
                    del model_inputs, generated_ids, output_ids, text
                    torch.cuda.empty_cache()

            print("\nEvaluation complete. Calculating metrics...")
            
            # Calculate Accuracy
            results_df = pd.read_csv(results_file)
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            UNKNOWN = 0

            for _, row in results_df.iterrows():
                y_true = str(row['label']).strip()
                y_pred = str(row['answer']).lower()

                # Use word boundary matching to ignore substrings (e.g. "no" in "unknown" or "know")
                pred_yes = re.search(r'\byes\b', y_pred) is not None
                pred_no = re.search(r'\bno\b', y_pred) is not None

                if y_true == "yes" and pred_yes and not pred_no:
                    TP += 1
                elif y_true == "no" and pred_no and not pred_yes:
                    TN += 1
                elif y_true == "no" and pred_yes and not pred_no:
                    FP += 1
                elif y_true == "yes" and pred_no and not pred_yes:
                    FN += 1
                else:
                    UNKNOWN += 1

            print(f"\nResults:")
            print(f"TP: {TP}  FP: {FP}")
            print(f"FN: {FN}  TN: {TN}")
            print(f"UNKNOWN: {UNKNOWN}/{len(results_df)}")
            accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
            print(f"Accuracy: {accuracy:.2f}%")

        except Exception as e:
            print(f"Error during benchmark evaluation: {e}")
            traceback.print_exc()

