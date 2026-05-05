# Quick Start

Back to [README](../README.md) · See also [evaluation.md](evaluation.md) and [openclaw-setup.md](openclaw-setup.md)

## Install

```bash
pip install -e .
pip install -e ".[dev]"
```

## Generate the benchmark snapshot

```bash
python scripts/generate_tasks.py
```

Generated outputs are written under `openclaw_env/data/{tasks,datasets}`.

## Run a first hard-benchmark evaluation

```bash
python examples/train_and_eval.py \
  --agent llm \
  --llm-provider openai \
  --llm-base-url https://api.example.com/v1 \
  --model Kimi-K2.5 \
  --task-prefix hard_decision_workflow_ \
  --split test \
  --mode multi \
  --max-steps 20 \
  --llm-max-tokens 192 \
  --llm-timeout-s 45 \
  --llm-request-retries 4 \
  --llm-retry-backoff-s 4 \
  --inter-task-sleep 3 \
  -v \
  --verbose-log ./tmp/hard_test_verbose.log \
  --save-report ./tmp/hard_test_report.json
```

This command uses a **recommended hard-benchmark protocol** (`--task-prefix hard_decision_workflow_ --mode multi --max-steps 20`) rather than the raw CLI defaults. The CLI default step budget is lower (`--max-steps 15`), so benchmark-facing runs should set the budget explicitly.

## Run Claude on AWS Bedrock through LiteLLM

Start a local LiteLLM proxy and export its credentials:

```bash
export LITELLM_PROXY_KEY="sk-litellm-master-key"
export LITELLM_PROXY_BASE_URL="http://127.0.0.1:4000/v1"
```

Then run:

```bash
python examples/train_and_eval.py \
  --agent llm \
  --llm-provider openai \
  --model claude-sonnet-4.6 \
  --task-prefix hard_decision_workflow_ \
  --split test \
  --mode multi \
  --max-steps 20 \
  --llm-max-tokens 1024 \
  -v
```

When `--llm-provider openai` is used with a `claude-*` model name, the client defaults to the local LiteLLM proxy URL and reads `LITELLM_PROXY_KEY` unless `--llm-base-url` or `--llm-api-key-env` is overridden.

The current default LLM rollout policy uses full interaction history. To override it, pass:

- `--llm-history-mode full`
- `--llm-history-mode summary`
- `--llm-history-mode auto`

## Common next steps

- For additional evaluation commands and output interpretation, see [evaluation.md](evaluation.md).
- For OpenClaw CLI, gateway, and provider bootstrap, see [openclaw-setup.md](openclaw-setup.md).
- For changing hard-scenario counts or regenerating different benchmark snapshots, see [task-generation.md](task-generation.md).
