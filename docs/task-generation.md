# Task Generation

Back to [README](../README.md) · See also [hard-benchmark.md](hard-benchmark.md) and [project-structure.md](project-structure.md)

## Output location

Generated outputs are written under:

- `openclaw_env/data/tasks`
- `openclaw_env/data/datasets`

## Regenerate the benchmark

```bash
python scripts/generate_tasks.py
```

## Hard benchmark generation knobs

The hard benchmark is generator-configurable. The released `362`-task hard snapshot is one official profile, not a hard-coded ceiling.

Two layers matter here:

- the **shared base count**, controlled by `--hard-decision-variants-per-scenario`
- **explicit per-scenario overrides**, controlled by `--hard-decision-scenario-counts`

If no explicit overrides are passed, unspecified hard scenarios fall back to the shared base count.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--hard-decision-variants-per-scenario <int>` | base hard count before applying the built-in scenario profile |
| `--hard-decision-scenario-counts a=INT,b=INT` | override specific hard-scenario counts |
| `--output-data-dir <path>` | write generated tasks to a custom root |
| `--complex-task-pack {off,standard}` | include or disable the secondary complex-workflow pack |

## Coverage reports

Generation writes coverage metadata to:

- [`openclaw_env/data/datasets/generator_coverage_report.json`](../openclaw_env/data/datasets/generator_coverage_report.json)
- [`openclaw_env/data/datasets/hard_split_coverage_report.json`](../openclaw_env/data/datasets/hard_split_coverage_report.json)

## Notes

- Hard-task metadata is stored in `TaskData.public`, including scenario name, ability tags, prompt style, and step count.
- Changing per-scenario counts changes the release profile, not the underlying scenario semantics.
- The generator script also exposes `--complex-task-pack`, `--complex-scenario-profile`, and optional branch-sensitive generation for auxiliary benchmark families.
