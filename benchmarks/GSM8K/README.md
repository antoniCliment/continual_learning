# GSM8K Next-Step Benchmark

This directory evaluates a model on GSM8K one reasoning step at a time. The
scripts should be run from this directory in the order below.

## Execution Order

1. Build the next-step benchmark files.

   ```bash
   python build_next_step_benchmark.py
   ```

   Inputs:
   - `train_socratic.jsonl`
   - `test_socratic.jsonl`

   Outputs:
   - `train_next_step.jsonl`
   - `test_next_step.jsonl`

2. Run the model on the next-step benchmark.

   ```bash
   python test_bench.py
   ```

   Useful options:
   - `--model-path`: model to load. Default:
     `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`.
   - `--bench-file`: benchmark JSONL to evaluate. Default:
     `test_next_step.jsonl`.
   - `--limit`: only evaluate the first N benchmark rows.
   - `--max-new-tokens`: generation length limit. Default: `256`.
   - `--thinking`: enable thinking mode if the tokenizer supports it.

   Outputs:
   - `results_<bench_stem>_<model_name>[_thinking].csv`
   - `metrics_<bench_stem>_<model_name>[_thinking].csv`

3. Optionally run Gemma as an LLM judge over the result CSV.

   ```bash
   python judge_results_with_gemma4.py
   ```

   The judge script auto-discovers a single `results_*.csv` in this directory.
   It does not currently expose a `--results-file` option, so if more than one
   exists, temporarily move or rename the other `results_*.csv` files.

   Useful options:
   - `--model-path`: judge model to load. Default:
     `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`.
   - `--candidate-column`: column to judge. Defaults to `extracted_answer` when
     present, otherwise `prediction`.
   - `--output-file`: judged CSV path. Default:
     `judge_<results_file_stem>.csv`.
   - `--metrics-file`: judge metrics path. Default:
     `judge_metrics_<results_file_stem>.csv`.
   - `--limit`: only judge the first N rows.
   - `--max-new-tokens`: judge generation length limit. Default: `192`.
   - `--thinking`: enable thinking mode if the tokenizer supports it.

4. Optionally create plots for judged result CSVs.

   ```bash
   python plot_judge_results.py
   ```

   By default, the plot script auto-discovers judge result CSV files in this
   directory that contain both `correct` and `gemma_judge_correct`. You can also
   pass one or more judged CSV paths explicitly.

   Useful options:
   - `csv_files`: optional positional list of judge result CSV files to plot.
   - `--output-dir`: directory for generated SVG files. Defaults to each input
     CSV's directory.

   Outputs:
   - `judge_plots_<judge_result_stem>.svg`

## Next-Step JSONL Variables

These variables are produced by `build_next_step_benchmark.py` in
`train_next_step.jsonl` and `test_next_step.jsonl`.

| Variable | Meaning |
| --- | --- |
| `id` | Unique benchmark item id, for example `test-00001-step-02`. |
| `split` | Source split name, usually `train` or `test`. |
| `source_file` | Socratic source file used to create the item. |
| `problem_index` | 1-based GSM8K problem number inside the source JSONL file. |
| `step_index` | 1-based reasoning step being evaluated for this problem. |
| `total_steps` | Total number of Socratic reasoning steps for the original problem. |
| `problem` | Original GSM8K word problem. |
| `previous_steps` | Solved step question/answer pairs that come before this item. |
| `next_step_question` | The Socratic question for the step the model must answer next. |
| `question` | Full prompt sent to the model, including problem, previous steps, and next step. |
| `answer` | Reference answer for only the requested next step, with GSM8K markers removed. |
| `raw_answer` | Reference step answer before removing GSM8K calculation markers. |
| `final_answer` | Original full-problem GSM8K final answer from the `####` line. |
| `is_final_step` | Boolean indicating whether this step is the last step for the problem. |
| `evaluation` | Object describing how the step should be scored. |
| `evaluation.kind` | Scoring mode: `numeric` or `text`. |
| `evaluation.target` | Exact value/text used as the scoring target. |
| `evaluation.source` | How the target was selected: `gsm8k_marker`, `last_number`, or `step_text`. |

## Result CSV Variables

These variables are produced by `test_bench.py` in `results_*.csv`.

| Variable | Meaning |
| --- | --- |
| `id` | Unique benchmark item id copied from the benchmark JSONL. |
| `problem_index` | 1-based GSM8K problem number from the source Socratic file. |
| `step_index` | 1-based step number evaluated for that problem. |
| `total_steps` | Total number of Socratic steps in the original problem. |
| `is_final_step` | `1` if this is the last step for the problem, otherwise `0`. |
| `next_step_question` | The step-level question the model was asked to answer. |
| `expected_answer` | Reference answer for only the requested next step. |
| `evaluation_kind` | Scoring mode from `evaluation.kind`: `numeric` or `text`. |
| `evaluation_source` | Target selection method from `evaluation.source`. |
| `evaluation_target` | Target used for scoring. For numeric rows, this is the expected number. For text rows, this is the expected text. |
| `prediction` | Raw generated text decoded from the model. |
| `extracted_answer` | Cleaned model output after removing wrappers such as code fences, `Answer:`, `Output:`, or hidden thinking prefixes. |
| `extracted_target` | Value compared to `evaluation_target`. For numeric rows, this is the last number extracted from `extracted_answer`; for text rows, this is reported as `text_f1=<score>`. |
| `correct` | `1` if the model answer matched the scoring target, otherwise `0`. |

`evaluation_source` values:

| Value | Meaning |
| --- | --- |
| `gsm8k_marker` | Target came from the last GSM8K marker like `<<calculation=result>>`. |
| `last_number` | Target came from the last number in the cleaned step answer because no GSM8K marker was available. |
| `step_text` | No numeric target was found, so the whole step answer is evaluated as text. |

## Metric CSV Variables

These variables are produced by `test_bench.py` in `metrics_*.csv`.

| Variable | Meaning |
| --- | --- |
| `group` | Metric grouping name: `overall`, `evaluation_source`, `step_index`, `step_type`, or `unknown`. |
| `key` | Bucket within the group, such as `all`, `gsm8k_marker`, a step index, `final`, `intermediate`, or `numeric_no_extraction`. |
| `correct` | Number of correct rows in the bucket. For `unknown/numeric_no_extraction`, this is the number of numeric rows where no number could be extracted. |
| `total` | Number of evaluated rows in the bucket. |
| `accuracy` | Percentage correct for the bucket, formatted from 0 to 100. This is blank for `unknown/numeric_no_extraction`. |

## Gemma Judge Result Variables

These variables are added by `judge_results_with_gemma4.py` in the judged CSV.
All original `results_*.csv` columns are preserved.

| Variable | Meaning |
| --- | --- |
| `gemma_judge_correct` | `1` if the judge marked the model answer correct, `0` if incorrect, and blank if the judge output could not be parsed. |
| `gemma_judge_reason` | Short explanation parsed from the judge response. |
| `gemma_judge_raw_output` | Raw judge model output before parsing. |

## Gemma Judge Metric Variables

These metric names are produced by `judge_results_with_gemma4.py` in
`judge_metrics_*.csv`, with columns `metric` and `value`.

| Metric | Meaning |
| --- | --- |
| `total` | Number of rows sent to the judge. |
| `judge_correct` | Number of rows the judge marked correct. |
| `judge_incorrect` | Number of rows the judge marked incorrect. |
| `judge_unknown` | Number of rows where the judge response could not be parsed into a boolean judgment. |
| `judge_accuracy_excluding_unknown` | `judge_correct / (total - judge_unknown) * 100`. |
| `compared_with_original_correct` | Number of rows where the original `correct` column could be parsed for agreement comparison. |
| `agreement_with_original_correct` | Number of comparable rows where `gemma_judge_correct` matched the original `correct` value. |
| `agreement_rate` | `agreement_with_original_correct / compared_with_original_correct * 100`. |

## Judge Plot Outputs

This file is produced by `plot_judge_results.py` from judged result CSVs.

| File | Meaning |
| --- | --- |
| `judge_plots_<judge_result_stem>.svg` | SVG dashboard comparing the benchmark scorer and Gemma judge. It includes overall rates, accuracy/agreement by step, agreement categories by step, and top evaluation-source groups. |

If the input CSV stem starts with `judge_results_`, that prefix is removed from
the plot output stem. For example, `judge_results_test_next_step_model.csv`
produces `judge_plots_test_next_step_model.svg`.
