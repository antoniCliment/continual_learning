from unsloth import FastVisionModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from datasets import Dataset
import torch
import csv
import os
import sys

from benchmark_utils import run_benchmark_evaluation, BenchmarkCallback

# ------------------------------------------------------------------------
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True   # Use 4bit quantization to reduce memory usage. Can be False.

# ------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python train.py [model_name] [lora_folder] [train_data_folder] [ba_bench_folder] [mc_bench_folder] [data_theme] [eval_steps]")
        sys.exit(1)
    
    model_id = sys.argv[1]
    lora_folder = sys.argv[2]
    train_data_folder = sys.argv[3]
    ba_bench_folder = sys.argv[4]
    mc_bench_folder = sys.argv[5]
    data_theme = sys.argv[6]
    eval_steps= sys.argv[7]
    eval_steps = int(eval_steps)
    lora_folder_name = lora_folder.split("/")[-1]

    print(f"Loading Unsloth model: {model_id} on data theme {data_theme} for {eval_steps} steps...")

    

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not hasattr(model.config, "pad_token_id"):
        setattr(model.config, "pad_token_id", tokenizer.pad_token_id)
    elif model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Baseline Evaluation
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    run_benchmark_evaluation(model, tokenizer, ba_bench_folder, mc_bench_folder, lora_folder_name, 0)
    print("="*60 + "\n")

    # Reset to training mode after baseline evaluation
    FastVisionModel.for_training(model)

    # Add LoRA adapters
    peft_args = {
        "finetune_vision_layers":     False, # False if not finetuning vision layers
        "finetune_language_layers":   True, # False if not finetuning language layers
        "finetune_attention_modules": True, # False if not finetuning attention layers
        "finetune_mlp_modules":       True, # False if not finetuning MLP layers
        "r": 16,
        #"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None,
    }
    model = FastVisionModel.get_peft_model(model, **peft_args)
    FastVisionModel.for_training(model)

    # Process  dataset
    formatted_samples = []
    csv_path = os.path.join(train_data_folder, "train.csv")
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) < 2: continue
            fact_text, text_type = row[1], row[0]
            start_marker, end_marker = "Text type: ", ". Characteristic:"
            if start_marker in text_type and end_marker in text_type:
                start = text_type.find(start_marker) + len(start_marker)
                end = text_type.find(end_marker)
                text_type_extracted = text_type[start:end].strip()
            else:
                text_type_extracted = "a description"
            messages = [
                {"role": "user", "content": f"Write {text_type_extracted} about {data_theme}."},
                {"role": "assistant", "content": fact_text}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            formatted_samples.append(text)

    dataset = Dataset.from_dict({"text": formatted_samples})

    # Training Arguments
    results_folder = os.path.join(ba_bench_folder, f"results_{lora_folder_name}")
    os.makedirs(results_folder, exist_ok=True)
    log_dir = os.path.join(results_folder, "logs")

    training_args = SFTConfig(
        output_dir = lora_folder,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.03,
        num_train_epochs = 3,
        learning_rate = 5e-6,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        packing = True, 
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        report_to = "wandb",
        logging_dir = log_dir,
        save_strategy = "steps",
        save_steps = 500,
    )

    benchmark_callback = BenchmarkCallback(ba_bench_folder, mc_bench_folder, tokenizer, lora_folder_name, eval_steps)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = training_args,
        callbacks=[benchmark_callback]
    )
    benchmark_callback.trainer = trainer

    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu_stats.name}. Max memory = {gpu_stats.total_memory / 1e9:.3f} GB.")

    print("Starting training...")
    trainer.train()

    model.save_pretrained(lora_folder) 
    tokenizer.save_pretrained(lora_folder)

    # Save training info
    with open(os.path.join(results_folder, "training_args.txt"), 'w') as f:
        f.write(f"PEFT Args: {peft_args}\n\nTraining Args: {training_args.to_dict()}\n")
    print("Training completed.")
