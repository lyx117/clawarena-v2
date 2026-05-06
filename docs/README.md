# Documentation

This directory contains the detailed documentation that used to live in the root README. The documents below are organized for benchmark users first: install the repo, run evaluations, understand the runtime mode, and then go deeper into benchmark design or generator configuration.

## Start Here

| Goal | Document |
| --- | --- |
| Install the repo and run a first benchmark command | [quickstart.md](quickstart.md) |
| Run `train_and_eval.py` and interpret logs/reports | [evaluation.md](evaluation.md) |
| Understand `mock`, `multi`, `real`, and `hybrid` | [execution-modes.md](execution-modes.md) |
| Set up OpenClaw, the gateway, and online providers | [openclaw-setup.md](openclaw-setup.md) |
| Regenerate tasks or change hard-scenario counts | [task-generation.md](task-generation.md) |
| Understand the hard benchmark design and scenario families | [hard-benchmark.md](hard-benchmark.md) |
| See the current documented release results and caveats | [results.md](results.md) |
| Orient yourself in the repo and development workflow | [project-structure.md](project-structure.md) |

## Plain-Language Overviews

- [ClawForge Data and Scenario Overview](clawforge-data-and-scenarios-overview.md) — a plain-English explanation of how V2 tasks are generated, how scenario families work, and what makes the benchmark hard
- [ClawForge 数据与场景概览（中文版）](clawforge-data-and-scenarios-overview.zh.md) — 上一篇文档的中文版本，用更直白的方式解释 V2 的数据生成、场景构造和 benchmark 难点

## Integration / Maintainer Guides

- [ClawForge Integration Guide](clawforge-integration-guide.md) — guidance for teams that need to place the current V2 benchmark line into a broader repo and write accurate umbrella-level documentation around it

## Historical / Maintainer Notes

- [Hard Benchmark Update Notes](HARD_BENCHMARK_UPDATE_NOTES.md) — archival notes for a prior benchmark update; not the canonical source for current release counts or current rollout defaults

Back to the repository landing page: [../README.md](../README.md)
