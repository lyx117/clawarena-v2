# Project Structure

Back to [README](../README.md) · See also [task-generation.md](task-generation.md)

## Repository map

```text
cli-agent-env/
├── docs/
├── examples/
├── openclaw_env/
│   ├── backend/
│   ├── core/
│   ├── data/
│   ├── evaluation/
│   ├── skills/
│   ├── tasks/
│   └── utils/
├── scripts/
├── tests/
└── README.md
```

## Where things live

- `examples/`: runnable entrypoints such as `train_and_eval.py`
- `openclaw_env/backend/`: backend implementations and execution routing
- `openclaw_env/core/`: environment and task abstractions
- `openclaw_env/data/`: generated tasks, splits, and coverage reports
- `openclaw_env/evaluation/`: checkers, metrics, and evaluator logic
- `openclaw_env/skills/`: app-family skills and adapters
- `openclaw_env/tasks/`: task generators, registry, and generation options
- `scripts/`: benchmark generation and utility scripts
- `tests/`: generator, runtime, and CLI regression tests

## Development checks

Targeted checks:

```bash
pytest -q tests/test_hard_decision_workflow_generator.py
pytest -q tests/test_train_and_eval_cli.py
pytest -q tests/test_hard_split_and_configs.py tests/test_generation_options.py
```

Broader suite:

```bash
python -m pytest tests/ -v
```

Regenerate benchmark artifacts before validating released counts or split reports:

```bash
python scripts/generate_tasks.py
```

## Safety summary

The environment applies two layers of protection:

- an allowlist for strict simulated backends
- a blocked-pattern filter for commands such as `rm -rf`, `sudo`, unrestricted `curl`/`wget`, and shell redirection

`real` and `hybrid` disable the allowlist for real `openclaw` execution, but blocked patterns still remain in force.
