import json
import re
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BENCHMARK_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BENCHMARK_DIR / "data" / "processed"
STEP_SEPARATOR = " ** "
STEP_SEPARATOR_RE = re.compile(r"\s*\*\*\s*", flags=re.ASCII)
MARKER_RE = re.compile(r"<<([^<>]+)>>")
NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?%?"
)


def strip_gsm8k_markers(text):
    return MARKER_RE.sub("", text).strip()


def extract_final_answer(answer):
    for line in answer.splitlines():
        line = line.strip()
        if line.startswith("####"):
            return line.removeprefix("####").strip()
    return ""


def extract_steps(answer):
    steps = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        if "**" not in line:
            raise ValueError(f"Step line does not contain {STEP_SEPARATOR!r}: {line}")
        step_question, step_answer = STEP_SEPARATOR_RE.split(line, 1)
        if not step_answer.strip():
            continue
        steps.append(
            {
                "question": step_question.strip(),
                "answer": strip_gsm8k_markers(step_answer),
                "raw_answer": step_answer.strip(),
            }
        )
    return steps


def normalize_number_text(value):
    return value.strip().replace("$", "").replace(",", "")


def last_marker_target(raw_answer):
    matches = MARKER_RE.findall(raw_answer)
    if not matches:
        return None

    marker = matches[-1]
    if "=" in marker:
        marker = marker.rsplit("=", 1)[1]

    numbers = NUMBER_RE.findall(marker)
    if numbers:
        return normalize_number_text(numbers[-1])

    return marker.strip()


def last_number_target(answer):
    numbers = NUMBER_RE.findall(answer)
    if not numbers:
        return None
    return normalize_number_text(numbers[-1])


def build_evaluation(step):
    marker_target = last_marker_target(step["raw_answer"])
    if marker_target is not None:
        return {
            "kind": "numeric",
            "target": marker_target,
            "source": "gsm8k_marker",
        }

    number_target = last_number_target(step["answer"])
    if number_target is not None:
        return {
            "kind": "numeric",
            "target": number_target,
            "source": "last_number",
        }

    return {
        "kind": "text",
        "target": step["answer"],
        "source": "step_text",
    }


def format_previous_steps(previous_steps):
    if not previous_steps:
        return "None yet."

    lines = []
    for idx, step in enumerate(previous_steps, start=1):
        lines.append(f"{idx}. {step['question']}")
        lines.append(f"   {step['answer']}")
    return "\n".join(lines)


def build_prompt(problem, previous_steps, next_step_question):
    return (
        "You are solving a math problem one step at a time.\n\n"
        "Problem:\n"
        f"{problem}\n\n"
        "Previous solved steps:\n"
        f"{format_previous_steps(previous_steps)}\n\n"
        "Next step:\n"
        f"{next_step_question}\n\n"
        "Solve only this next step. Do not solve later steps. "
        "Keep the answer to one short sentence and include the calculation when needed. "
        "When there is a calculation, end with the value for this next step."
    )


def convert_file(source_path, output_path, split_name):
    total_steps = 0
    with source_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for problem_index, line in enumerate(source, start=1):
            record = json.loads(line)
            problem = record["question"]
            final_answer = extract_final_answer(record["answer"])
            steps = extract_steps(record["answer"])

            for step_index, step in enumerate(steps, start=1):
                previous_steps = steps[: step_index - 1]
                item = {
                    "id": f"{split_name}-{problem_index:05d}-step-{step_index:02d}",
                    "split": split_name,
                    "source_file": source_path.name,
                    "problem_index": problem_index,
                    "step_index": step_index,
                    "total_steps": len(steps),
                    "problem": problem,
                    "previous_steps": [
                        {
                            "question": previous_step["question"],
                            "answer": previous_step["answer"],
                        }
                        for previous_step in previous_steps
                    ],
                    "next_step_question": step["question"],
                    "question": build_prompt(
                        problem,
                        previous_steps,
                        step["question"],
                    ),
                    "answer": step["answer"],
                    "raw_answer": step["raw_answer"],
                    "final_answer": final_answer,
                    "is_final_step": step_index == len(steps),
                    "evaluation": build_evaluation(step),
                }
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_steps += 1
    return total_steps


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    conversions = [
        (
            "train",
            RAW_DATA_DIR / "train_socratic.jsonl",
            PROCESSED_DATA_DIR / "train_next_step.jsonl",
        ),
        (
            "test",
            RAW_DATA_DIR / "test_socratic.jsonl",
            PROCESSED_DATA_DIR / "test_next_step.jsonl",
        ),
    ]

    for split_name, source_path, output_path in conversions:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        total_steps = convert_file(source_path, output_path, split_name)
        print(f"Wrote {total_steps} next-step items to {output_path}")


if __name__ == "__main__":
    main()
