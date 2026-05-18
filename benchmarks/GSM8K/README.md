# GSM8K Next-Step Benchmark

This benchmark evaluates a model on GSM8K one reasoning step at a time.
Examples below assume your current directory is `benchmarks/GSM8K`.

## Layout

- `scripts/`: benchmark builders, evaluators, judge runner, and plotting tools.
- `data/raw/`: source GSM8K Socratic JSONL files.
- `data/processed/`: generated next-step benchmark JSONL files.
- `results/`: evaluator CSVs and judge CSVs.
- `plots/`: generated SVG dashboards.
- `logs/`: captured run logs.
- `cache/`: local Python and Unsloth cache artifacts.

## Workflow

1. Build the next-step benchmark files.

   ```bash
   python3 scripts/build_next_step_benchmark.py
   ```

   Inputs:
   - `data/raw/train_socratic.jsonl`
   - `data/raw/test_socratic.jsonl`

   Outputs:
   - `data/processed/train_next_step.jsonl`
   - `data/processed/test_next_step.jsonl`

2. Run the model on the next-step benchmark.

   ```bash
   python3 scripts/test_bench.py
   ```

   Useful options:
   - `--model-path`: model to load. Default:
     `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`.
   - `--bench-file`: benchmark JSONL to evaluate. Default:
     `data/processed/test_next_step.jsonl`.
   - `--output-dir`: destination for CSV outputs. Default: `results/`.
   - `--limit`: only evaluate the first N benchmark rows.
   - `--max-new-tokens`: generation length limit. Default: `256`.
   - `--thinking`: enable thinking mode if the tokenizer supports it.

   Outputs:
   - `results/results_<bench_stem>_<model_name>[_thinking].csv`
   - `results/metrics_<bench_stem>_<model_name>[_thinking].csv`

3. Optionally run Gemma as an LLM judge over a result CSV.

   ```bash
   python3 scripts/judge_results_with_gemma4.py
   ```

   By default, the judge auto-discovers a single `results_*.csv` in `results/`.
   If there is more than one, pass the one to judge explicitly:

   ```bash
   python3 scripts/judge_results_with_gemma4.py --results-file results/results_test_next_step_model.csv
   ```

   Useful options:
   - `--results-file`: result CSV to judge.
   - `--model-path`: judge model to load. Default:
     `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`.
   - `--candidate-column`: column to judge. Defaults to `extracted_answer` when
     present, otherwise `prediction`.
   - `--output-dir`: destination for judge CSV outputs. Default: `results/`.
   - `--output-file`: explicit judged CSV path.
   - `--metrics-file`: explicit judge metrics CSV path.
   - `--limit`: only judge the first N rows.
   - `--max-new-tokens`: judge generation length limit. Default: `192`.
   - `--thinking`: enable thinking mode if the tokenizer supports it.

   Outputs:
   - `results/judge_<results_file_stem>.csv`
   - `results/judge_metrics_<results_file_stem>.csv`

4. Optionally create plots for judged result CSVs.

   ```bash
   python3 scripts/plot_judge_results.py
   ```

   By default, the plot script reads judged CSVs in `results/` that contain both
   `correct` and `gemma_judge_correct`, then writes SVGs to `plots/`.

   Useful options:
   - `csv_files`: optional positional list of judged CSV paths to plot.
   - `--output-dir`: directory for generated SVG files. Default: `plots/`.

   Outputs:
   - `plots/judge_plots_<judge_result_stem>.svg`

## File Schemas

`data/processed/*_next_step.jsonl` is produced by
`scripts/build_next_step_benchmark.py`.

| Variable | Meaning |
| --- | --- |
| `id` | Unique benchmark item id, for example `test-00001-step-02`. |
| `split` | Source split name, usually `train` or `test`. |
| `source_file` | Raw Socratic source file used to create the item. |
| `problem_index` | 1-based GSM8K problem number inside the source JSONL file. |
| `step_index` | 1-based reasoning step being evaluated for this problem. |
| `total_steps` | Total number of Socratic reasoning steps for the original problem. |
| `problem` | Original GSM8K word problem. |
| `previous_steps` | Solved step question/answer pairs that come before this item. |
| `next_step_question` | The Socratic question for the step the model must answer next. |
| `question` | Full prompt sent to the model. |
| `answer` | Reference answer for only the requested next step. |
| `raw_answer` | Reference step answer before removing GSM8K calculation markers. |
| `final_answer` | Original full-problem GSM8K final answer from the `####` line. |
| `is_final_step` | Whether this step is the last step for the problem. |
| `evaluation` | Scoring metadata with `kind`, `target`, and `source`. |

`results/results_*.csv` is produced by `scripts/test_bench.py`.

| Variable | Meaning |
| --- | --- |
| `id` | Benchmark item id copied from the JSONL. |
| `problem_index`, `step_index`, `total_steps`, `is_final_step` | Step metadata copied from the JSONL. |
| `next_step_question` | Step-level question the model answered. |
| `expected_answer` | Reference answer for only the requested next step. |
| `evaluation_kind`, `evaluation_source`, `evaluation_target` | Scoring metadata copied from `evaluation`. |
| `prediction` | Raw generated text decoded from the model. |
| `extracted_answer` | Cleaned model output after removing wrappers or hidden thinking prefixes. |
| `extracted_target` | Numeric target extracted from the answer, or text F1 for text rows. |
| `correct` | `1` if the model answer matched the scoring target, otherwise `0`. |

`results/metrics_*.csv` is also produced by `scripts/test_bench.py`.

| Variable | Meaning |
| --- | --- |
| `group` | Metric grouping name. |
| `key` | Bucket within the group. |
| `correct` | Number of correct rows in the bucket. |
| `total` | Number of evaluated rows in the bucket. |
| `accuracy` | Percentage correct for the bucket. |

`results/judge_*.csv` is produced by `scripts/judge_results_with_gemma4.py`.
All original `results_*.csv` columns are preserved, with these additions:

| Variable | Meaning |
| --- | --- |
| `gemma_judge_correct` | `1` if the judge marked the model answer correct, `0` if incorrect, blank if unparsed. |
| `gemma_judge_reason` | Short explanation parsed from the judge response. |
| `gemma_judge_raw_output` | Raw judge model output before parsing. |
