# Industrial Time-Series Adapter

This adapter converts the time-series anomaly benchmark into
Terminal-Bench task directories. It supports the original layout in
`data/.../scenario_*/rec_*` as well as the flattened
`exam_questions/<model>/` exports produced by `workflows/create_exam_questions.py`
that bundle question dataframes and baselines.

## Features

- Copies `data.pkl` into the task container so agents can analyse the
  raw time-series directly.
- Generates `task.yaml` instructions with lettered choices and clear
  delivery requirements for `results.json`.
- Stores the correct answer inside `tests/ground_truth.json` and checks
  it with `tests/test_results.py`.
- Produces a sanitized `scenario_info.json` that omits
  answer-revealing fields from the original metadata.
- Optional extras via CLI flags: copy preview images, copy diagram
  folders, or export a `data_preview.csv` with the first *N* rows.

## Usage

```bash
python terminal-bench/adapters/industrial_timeseries/run_adapter.py \
    # defaults to exam_questions/ auto-discovery, so no path needed
    --limit 5 \
    --include-preview \
    --csv-sample-rows 500
```

To target a specific root, you can still pass it (not required):

```bash
python terminal-bench/adapters/tsENV/run_adapter.py \
    exam_questions/DoubleMassSpringDamper \
    --limit 5 \
    --output-dir terminal-bench/tasks/simbench \
    --task-prefix simbench \
    --prefix-model
```

Key flags:

- `--limit`: cap the number of converted `rec_*` directories.
- `--only`: pass explicit relative identifiers if you want a targeted
  conversion.
- `--contains`: include only instances whose relative path contains the
  provided substring.
- `--prefix-model` / `--no-prefix-model`: include (default) or skip the
  model name in task IDs (default layout: `simbench/<model>-scenario-...`).
- `--include-preview` / `--include-diagrams`: copy optional visual
  assets from the source dataset.
- `--csv-sample-rows`: generate `data_preview.csv` with the first *N*
  rows (requires `pandas` during adapter execution).

Run with `--dry-run` first to preview which tasks would be created.

Each generated task lives under the chosen output directory and
contains the standard Terminal-Bench files (`Dockerfile`,
`task.yaml`, `run-tests.sh`, `solution.sh`, and a `tests/`
subdirectory). The verification harness expects the agent to write a
`results.json` file. For `simbench_cls` classification tasks it must contain
`{"predictions": {"test_samples/<filename>": {"change_time": <t_or_index>, "final_answer": "<class_label>"}, ...}}`.
For tsENV direct and open-ended tasks, follow the per-sample `results.json`
shape described in the generated task prompt. For other classification tasks it
must contain `{"final_answer": "<class_label>"}`.
