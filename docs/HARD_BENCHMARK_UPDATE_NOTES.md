# Hard Benchmark Update Notes

> Historical note: this file records a prior benchmark update and intentionally preserves the counts, defaults, and framing from that milestone. For current release counts, current hard-benchmark docs, and current rollout defaults, use the root [README](../README.md), [hard-benchmark.md](hard-benchmark.md), and [evaluation.md](evaluation.md).

This note summarizes the benchmark update relative to a prior internal baseline snapshot.

## Benchmark summary

The benchmark has shifted from a generic multi-domain task collection toward a more explicit flagship benchmark:

- `hard_decision_workflow` is now the primary benchmark family.
- The benchmark center is an **interactive, step-by-step, real-execution-oriented** hard decision suite.
- Agents are expected to inspect state, act through CLI commands, and finish the workflow in an interactive environment.

Current hard benchmark facts:

- `17` hard scenarios
- `294` hard tasks
- formal comparison convention:
  - `--task-prefix hard_decision_workflow_`
  - `--mode multi`
  - `--max-steps 20`

## Key changes since the prior baseline snapshot

### 1. Hard benchmark core

Primary files:

- `openclaw_env/tasks/generators/hard_decision_workflows.py`
- `tests/test_hard_decision_workflow_generator.py`

What changed:

- large hard benchmark refactor around `hard_decision_workflow`
- explicit `17`-scenario hard profile and `294`-task default mix
- scenario-level metadata and ability tags in `TaskData.public`
- stratified split coverage for hard tasks
- fairness fixes:
  - `existing_state_followthrough` state-construction bug fixed
  - `inbox_followthrough` hidden calendar-title naming constraint removed
- instruction naturalness polish for the most templated release-family scenarios
- expanded regression coverage around counts, fairness, and wording

### 2. LLM evaluation and provider handling

Primary files:

- `examples/train_and_eval.py`
- `tests/test_train_and_eval_cli.py`

What changed:

- provider-noise accounting (`provider_impacted_tasks`)
- compact memory / trimmed request construction for OpenAI-compatible models
- GPT-5 token-field compatibility via `max_completion_tokens`
- more explicit hard-eval reporting and handling for provider-impacted runs

### 3. Runtime correctness and interactive execution fidelity

Primary files:

- `openclaw_env/core/task.py`
- `openclaw_env/core/environment.py`
- `openclaw_env/backend/mock_backend.py`
- `openclaw_env/backend/real_openclaw_backend.py`
- `openclaw_env/backend/openclaw_compat.py`
- `openclaw_env/backend/multi_app_backend.py`
- `openclaw_env/backend/tasks_backend.py`

What changed:

- tighter cron/message/channel parameter validation
- safer compat rewrite behavior for incomplete `openclaw cron add` commands
- stricter effect-state extraction in the real backend to avoid fake state changes
- tasks backend migrated to `TasksSkill` so runtime behavior is closer to the interactive command model used in evaluation
- task schema/public metadata support expanded for hard benchmark reporting

### 4. Generated benchmark data

Generated artifacts that matter for reproducibility:

- `openclaw_env/data/datasets/train.txt`
- `openclaw_env/data/datasets/dev.txt`
- `openclaw_env/data/datasets/test.txt`
- `openclaw_env/data/datasets/hard_split_coverage_report.json`
- `openclaw_env/data/datasets/generator_coverage_report.json`
- `openclaw_env/data/tasks/hard_decision_workflow_*/specs.json`

## Recommended push scope

### Must include

Benchmark core:

- `README.md`
- `openclaw_env/tasks/generators/hard_decision_workflows.py`
- `tests/test_hard_decision_workflow_generator.py`
- `examples/train_and_eval.py`
- `tests/test_train_and_eval_cli.py`

Runtime support:

- `openclaw_env/core/task.py`
- `openclaw_env/core/environment.py`
- `openclaw_env/backend/mock_backend.py`
- `openclaw_env/backend/real_openclaw_backend.py`
- `openclaw_env/backend/openclaw_compat.py`
- `openclaw_env/backend/multi_app_backend.py`
- `openclaw_env/backend/tasks_backend.py`

Generated data:

- `openclaw_env/data/datasets/train.txt`
- `openclaw_env/data/datasets/dev.txt`
- `openclaw_env/data/datasets/test.txt`
- `openclaw_env/data/datasets/hard_split_coverage_report.json`
- `openclaw_env/data/datasets/generator_coverage_report.json` if referenced in docs
- `openclaw_env/data/tasks/hard_decision_workflow_*/specs.json`

Optional docs:

- `HARD_MODEL_RESULTS_SUMMARY.md`
- `docs/HARD_BENCHMARK_UPDATE_NOTES.md`

### Do not include

- `tmp/`
- `__pycache__/`
- local verbose logs and reports
- scratch analysis files unless intentionally promoted to docs
- unrelated non-hard task churn

## README framing for this push

The README should make three points early and explicitly:

1. this is an **interactive CLI-agent benchmark**
2. the environment is built around **real step-by-step execution**
3. the current flagship benchmark is **`hard_decision_workflow`**

General benchmark totals, complex workflow generation, and auxiliary task families should remain documented, but not as the opening story.

## Verification checklist

Run before push:

```bash
pytest -q tests/test_hard_decision_workflow_generator.py
pytest -q tests/test_train_and_eval_cli.py
pytest -q tests/test_hard_split_and_configs.py tests/test_generation_options.py
python scripts/generate_tasks.py
```

Then check:

- hard family remains `294` tasks
- hard split counts remain stratified
- staged files do not include `tmp/`, `__pycache__/`, or unrelated task churn
