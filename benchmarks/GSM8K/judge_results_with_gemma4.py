import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


# Keep this consistent with test_bench.py.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")


MODEL_PATH = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_GLOB = "results_*.csv"
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 192


def safe_model_name(model_path):
    return os.path.basename(os.path.normpath(model_path)).replace("/", "-")


def discover_results_file():
    matches = sorted(SCRIPT_DIR.glob(RESULTS_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No results CSV found in {SCRIPT_DIR} matching {RESULTS_GLOB}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Expected exactly one results CSV in "
            f"{SCRIPT_DIR}, found {len(matches)}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def load_rows(path, limit=None):
    with open(path, "r", newline="", encoding="utf-8") as results:
        reader = csv.DictReader(results)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no CSV header")

        required_columns = {"id", "next_step_question", "expected_answer"}
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_columns)}"
            )

        rows = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return reader.fieldnames, rows


def choose_candidate_column(fieldnames, requested_column):
    if requested_column:
        if requested_column not in fieldnames:
            raise ValueError(f"Candidate column not found: {requested_column}")
        return requested_column

    if "extracted_answer" in fieldnames:
        return "extracted_answer"
    if "prediction" in fieldnames:
        return "prediction"

    raise ValueError(
        "Could not find a generated-answer column. Expected extracted_answer or prediction."
    )


def build_judge_prompt(row, candidate_column):
    generated_answer = row.get(candidate_column, "")
    scoring_context = ""
    if row.get("evaluation_kind") or row.get("evaluation_target"):
        scoring_context = (
            "\nScoring information from the benchmark:\n"
            f"Kind: {row.get('evaluation_kind', '')}\n"
            f"Target: {row.get('evaluation_target', '')}\n"
        )

    return (
        "You are grading one step of a GSM8K math solution.\n\n"
        "Step question:\n"
        f"{row.get('next_step_question', '')}\n\n"
        "Reference answer:\n"
        f"{row.get('expected_answer', '')}\n\n"
        f"{scoring_context}"
        "Model answer:\n"
        f"{generated_answer}\n\n"
        "Decide whether the model answer is correct for this step.\n"
        "Grade semantic and mathematical equivalence, not exact wording.\n"
        "Ignore harmless formatting differences, LaTeX syntax, currency symbols, "
        "and unit phrasing when the meaning is unchanged.\n"
        "Mark incorrect if the answer gives the wrong value, omits the needed result, "
        "contradicts itself, answers a different step, or continues to a later step "
        "and ends with a later-step result.\n\n"
        "Return only valid JSON in this exact shape:\n"
        '{"correct": true, "reason": "short reason"}'
    )


def clean_judge_output(text):
    cleaned = str(text).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def parse_judgment(text):
    cleaned = clean_judge_output(text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and isinstance(parsed.get("correct"), bool):
            return parsed["correct"], str(parsed.get("reason", "")).strip()
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict) and isinstance(parsed.get("correct"), bool):
                return parsed["correct"], str(parsed.get("reason", "")).strip()
        except json.JSONDecodeError:
            pass

    bool_match = re.search(r'"?correct"?\s*:\s*(true|false)', cleaned, flags=re.I)
    if bool_match:
        return bool_match.group(1).lower() == "true", ""

    yes_no_match = re.search(r"\b(yes|no|correct|incorrect)\b", cleaned, flags=re.I)
    if yes_no_match:
        token = yes_no_match.group(1).lower()
        return token in {"yes", "correct"}, ""

    return None, ""


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


def parse_original_correct(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "correct"}:
        return True
    if text in {"0", "false", "no", "incorrect"}:
        return False
    return None


def write_metrics(path, total, judged_correct, unknown, agreements, comparable):
    judged_incorrect = total - judged_correct - unknown
    accuracy = (judged_correct / (total - unknown)) * 100 if total > unknown else 0
    agreement = (agreements / comparable) * 100 if comparable else 0

    with open(path, "w", newline="", encoding="utf-8") as metrics:
        writer = csv.writer(metrics)
        writer.writerow(["metric", "value"])
        writer.writerow(["total", total])
        writer.writerow(["judge_correct", judged_correct])
        writer.writerow(["judge_incorrect", judged_incorrect])
        writer.writerow(["judge_unknown", unknown])
        writer.writerow(["judge_accuracy_excluding_unknown", f"{accuracy:.2f}"])
        writer.writerow(["compared_with_original_correct", comparable])
        writer.writerow(["agreement_with_original_correct", agreements])
        writer.writerow(["agreement_rate", f"{agreement:.2f}"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use Gemma 4 as an LLM judge for a GSM8K next-step results CSV."
        )
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--candidate-column", default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode when the tokenizer chat template supports it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the results file and print the first judge prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        args.results_file = discover_results_file()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        sys.exit(1)

    if not args.results_file.exists():
        print(f"Results file not found: {args.results_file}")
        sys.exit(1)

    fieldnames, rows = load_rows(args.results_file, limit=args.limit)
    if not rows:
        print(f"No rows found in {args.results_file}")
        sys.exit(1)

    candidate_column = choose_candidate_column(fieldnames, args.candidate_column)

    model_name = safe_model_name(args.model_path)
    output_file = args.output_file or args.results_file.with_name(
        f"judge_{args.results_file.stem}_{model_name}.csv"
    )
    metrics_file = args.metrics_file or args.results_file.with_name(
        f"judge_metrics_{args.results_file.stem}_{model_name}.csv"
    )

    if args.dry_run:
        print(f"Loaded {len(rows)} rows from {args.results_file}")
        print(f"Candidate column: {candidate_column}")
        print(f"Output file: {output_file}")
        print(f"Metrics file: {metrics_file}")
        print("\nFirst judge prompt:\n")
        print(build_judge_prompt(rows[0], candidate_column))
        return

    import torch
    from unsloth import FastLanguageModel

    print(f"Loading judge model: {args.model_path}")
    print(f"Results file: {args.results_file}")
    print(f"Candidate column: {candidate_column}")
    print(f"Thinking mode: {'enabled' if args.thinking else 'disabled'}")

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

    output_fieldnames = list(fieldnames)
    for column in [
        "gemma_judge_correct",
        "gemma_judge_reason",
        "gemma_judge_raw_output",
    ]:
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    judged_correct = 0
    unknown = 0
    agreements = 0
    comparable = 0

    with torch.inference_mode(), open(
        output_file, "w", newline="", encoding="utf-8"
    ) as judged_results:
        writer = csv.DictWriter(judged_results, fieldnames=output_fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            prompt = build_judge_prompt(row, candidate_column)
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
            raw_output = tokenizer.decode(
                output_ids,
                skip_special_tokens=not args.thinking,
            ).strip()

            judgment, reason = parse_judgment(raw_output)
            if judgment is None:
                unknown += 1
                row["gemma_judge_correct"] = ""
            else:
                judged_correct += int(judgment)
                row["gemma_judge_correct"] = int(judgment)

                original_correct = parse_original_correct(row.get("correct", ""))
                if original_correct is not None:
                    comparable += 1
                    agreements += int(original_correct == judgment)

            row["gemma_judge_reason"] = reason
            row["gemma_judge_raw_output"] = raw_output
            writer.writerow(row)

            print(f"Judged {idx}/{len(rows)}", end="\r")

            del model_inputs, generated_ids, output_ids, text
            clear_accelerator_cache(torch)

    print()
    write_metrics(
        metrics_file,
        total=len(rows),
        judged_correct=judged_correct,
        unknown=unknown,
        agreements=agreements,
        comparable=comparable,
    )

    judged_total = len(rows) - unknown
    judge_accuracy = (judged_correct / judged_total) * 100 if judged_total else 0
    agreement_rate = (agreements / comparable) * 100 if comparable else 0

    print(f"Judged results file: {output_file}")
    print(f"Judge metrics file: {metrics_file}")
    print(
        f"Gemma judge accuracy: {judge_accuracy:.2f}% "
        f"({judged_correct}/{judged_total}, unknown={unknown})"
    )
    if comparable:
        print(
            f"Agreement with original correct column: {agreement_rate:.2f}% "
            f"({agreements}/{comparable})"
        )


if __name__ == "__main__":
    main()
