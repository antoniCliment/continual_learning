
from unsloth import FastLanguageModel, is_bfloat16_supported
import argparse
import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = True


def load_train_dataset(csv_path: str, tokenizer) -> Dataset:
    """
    Loads train.csv for SFT.

    Expected columns, in priority order:
    1. text
       - Used directly if present and non-empty.
    2. input + output
       - Converted into a chat-format training sample.
    """

    df = pd.read_csv(csv_path)

    if "text" in df.columns and df["text"].notna().any():
        texts = df["text"].dropna().astype(str).tolist()

    elif {"input", "output"}.issubset(df.columns):
        texts = []
        for _, row in df.dropna(subset=["input", "output"]).iterrows():
            messages = [
                {"role": "user", "content": str(row["input"])},
                {"role": "assistant", "content": str(row["output"])},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

    else:
        raise ValueError(
            "train.csv must contain either a non-empty 'text' column "
            "or both 'input' and 'output' columns."
        )

    if not texts:
        raise ValueError("No valid training rows found in train.csv.")

    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        default="train.csv",
        help="Path to train.csv. Defaults to ./train.csv",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Base model name or local path.",
    )
    parser.add_argument(
        "--output_dir",
        default="lora_model",
        help="Where to save the LoRA adapter and tokenizer.",
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading training data from: {args.train_csv}")
    train_dataset = load_train_dataset(args.train_csv, tokenizer)
    print(f"Training samples: {len(train_dataset)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        packing=True,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        report_to="none",
        save_strategy="steps",
        save_steps=args.save_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"GPU = {gpu_stats.name}. Max memory = {gpu_stats.total_memory / 1e9:.3f} GB.")

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapter and tokenizer to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Fine-tuning completed.")


if __name__ == "__main__":
    main()
