# ClawArena-V2 Integration Guide

Back to [README](../README.md) · See also [docs/README.md](README.md), [v2-data-and-scenarios-explained.md](v2-data-and-scenarios-explained.md), [hard-benchmark.md](hard-benchmark.md), [evaluation.md](evaluation.md), and [project-structure.md](project-structure.md)

## Purpose

This document is a detailed introduction to **ClawArena-V2** for developers who did not build it originally but now need to:

- understand what V2 is actually claiming
- place it correctly inside a larger repository or documentation set
- know which modules are core to V2
- know which parts are generic infrastructure and which parts are benchmark-specific
- write accurate umbrella-level documentation without diluting or mislabeling the benchmark

This is not a user quickstart and not a paper abstract. It is an integration-facing technical overview.

## Executive summary

ClawArena-V2 is an **interactive execution benchmark for CLI-style agents** operating over persistent software state.

Its central evaluation question is:

> Can an agent inspect evolving state, issue commands step by step, and complete a workflow through correct state changes and observable side effects?

That is different from a static text benchmark, where a model only needs to emit one final answer. In ClawArena-V2:

- the environment is stateful
- the agent acts command by command
- backends mutate after each step
- evaluation is result-first and state-aware
- multiple trajectories can succeed if they produce the same evaluated outcome

The current flagship suite is `hard_decision_workflow`, a generator-backed benchmark family centered on difficult workflow decisions such as partial completion, branch resolution, state repair, replacement, duplicate avoidance, and workflow closure.

## Installation and first-run path

Another development team usually needs more than a conceptual explanation. They also need the minimal operational path that proves V2 is concrete and runnable.

### Editable installation

From the repository root:

```bash
pip install -e .
pip install -e ".[dev]"
```

This installs the environment plus development dependencies used by generation scripts and tests.

### Regenerating benchmark artifacts

V2 benchmark assets are largely generated rather than manually edited. The standard regeneration command is:

```bash
python scripts/generate_tasks.py
```

This refreshes artifacts under:

- `openclaw_env/data/tasks/`
- `openclaw_env/data/datasets/`

If another team is integrating V2 into a broader repo, this command is the canonical way to rebuild release artifacts before validating counts, split files, or coverage reports.

### First evaluation run

The main evaluation entrypoint is:

```bash
python examples/train_and_eval.py
```

A representative hard-benchmark command is:

```bash
python examples/train_and_eval.py \
  --agent llm \
  --task-prefix hard_decision_workflow_ \
  --split test \
  --mode multi \
  --max-steps 20
```

The CLI default step budget is `15`, but benchmark-facing runs should usually set `--max-steps` explicitly. In current practice:

- `20` is the common protocol for standard hard runs
- `25` is sometimes used for manuscript-style full-history comparisons

### Hosted OpenAI-compatible example

One common hosted setup is:

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
  -v
```

### Claude via LiteLLM proxy

The current repo also supports a local LiteLLM proxy path for Claude-style models. In practice:

- run a local LiteLLM proxy
- point `--llm-provider openai` at that proxy
- use a `claude-*` model name

The detailed operational steps belong in [quickstart.md](quickstart.md) and [evaluation.md](evaluation.md), but another team documenting V2 should know this path exists because it affects how benchmark comparisons are actually run.

## OpenClaw and runtime prerequisites

ClawArena-V2 can run under several execution modes, and setup expectations depend on the mode.

### Modes that do not require a full OpenClaw deployment

- `mock`
- `multi`

These are the easiest modes for benchmark development and reproducible evaluation. `multi` is the benchmark-default mode.

### Modes that require real OpenClaw-backed setup

- `real`
- `hybrid`

These require a working OpenClaw installation and provider bootstrap. In practice that means:

- installing the `openclaw` CLI
- starting the gateway
- setting up Google-backed providers where Calendar, Gmail, or Tasks are involved
- exporting the required provider environment variables

Another team integrating V2 should understand that OpenClaw setup is shared runtime infrastructure, while the benchmark semantics remain V2-specific.

## Operational details that matter during integration

### History modes

The LLM rollout client currently supports three history policies:

- `full`
- `summary`
- `auto`

The current default is `full`.

This changes how much interaction history is sent back to the model during rollout. It does not change the task definition or evaluator contract, but it materially affects rollout behavior and should therefore be documented in evaluation-facing material.

### Useful output and reporting flags

When V2 is run through `examples/train_and_eval.py`, the most useful additional outputs are:

- `--verbose-log PATH`
- `--save-report PATH`
- `--llm-request-retries N`
- `--llm-retry-backoff-s S`
- `--inter-task-sleep S`

These matter in integration contexts because another team will often want:

- raw rollout traces
- JSON reports for downstream aggregation
- provider-aware accounting for unstable interactive endpoints

### Development checks

The most useful targeted checks for V2 are:

```bash
pytest -q tests/test_hard_decision_workflow_generator.py
pytest -q tests/test_train_and_eval_cli.py
pytest -q tests/test_hard_split_and_configs.py tests/test_generation_options.py
```

If another team is validating an integration branch, these are the first checks they should run before attempting a larger benchmark comparison.

## One-paragraph description for external use

If another team needs a short, accurate description for an umbrella README or repo introduction, use wording close to this:

> ClawArena-V2 is an interactive benchmark for command-line agents operating over persistent software state. Rather than scoring a final text answer, it evaluates whether an agent can inspect environment state, issue commands step by step, and complete workflows through real state changes and observable side effects. The benchmark is generator-backed, result-first, and evaluated through normalized execution state rather than exact action-sequence matching.

That wording matches the actual V2 contract and avoids importing claims that belong to other systems.

## What V2 is and is not

### What V2 is

ClawArena-V2 is:

- an interactive benchmark line
- a stateful execution environment for evaluation
- a generator-backed task suite
- a result-first evaluator over normalized state
- a benchmark for multi-step workflow execution

### What V2 is not

ClawArena-V2 is not:

- a static QA benchmark
- a one-shot text generation task
- a generic tool-use demo
- a training or continual-learning method
- a pure prompting framework
- an exact-trajectory imitation benchmark

The repository may contain generic runtime components that are reusable elsewhere, but the V2 contribution itself is about **interactive execution benchmarking under evolving state**.

## The core V2 design

ClawArena-V2 is organized around three high-level design choices.

### 1. Interactive execution instead of final-answer scoring

In a V2 rollout, the agent does not see one prompt and respond once. Instead:

1. the agent receives a task instruction
2. it inspects the current observation and visible environment state
3. it emits one command
4. the environment executes that command
5. the environment returns updated observation and state
6. the loop repeats until success or the step budget is exhausted

This means the benchmark can expose failures that static answer matching hides:

- repeated or duplicate work
- stale-state reasoning
- partial completion without closure
- wrong-state replacement
- branching errors under conflicting evidence

### 2. Automated generation instead of one fixed authored list

The benchmark is not defined as one immutable table of handcrafted tasks. It is produced from parameterized scenario families and generation logic.

At a high level, generation combines:

- scenario templates
- grounded slots and scenario-specific variables
- initial-state setup programs
- reference trajectories
- executable validators
- structured metadata

This is why current released counts such as:

- `17` hard scenarios
- `362` hard tasks
- `1616` total tasks

should be read as **current official release statistics**, not as hard-coded benchmark limits.

In other words, V2 should be described as a **benchmark family with an official release profile**, not as a single frozen CSV.

### 3. Functional evaluation over normalized state

V2 does not require one exact privileged action sequence.

Instead, the environment builds a normalized evaluator-facing state and runs typed result-first checks over it. This is what allows:

- different valid command sequences to pass
- evaluation to focus on actual environment outcomes
- partial-credit scoring to coexist with exact pass/fail checks

The key idea is that the evaluator reasons over what the agent *caused*, not merely over what text it *said*.

## The V2 runtime contract

If another team is integrating V2, they should understand the runtime contract, because this is where V2 differs most sharply from text-only benchmarks.

### Inputs

At task start, the agent is given:

- one task instruction
- an initial environment state
- a set of command families / skills / backends available in the current mode

### Per-step behavior

At each step, the agent:

- observes the latest environment output
- emits one command
- triggers execution through the routed backend layer

The environment then:

- executes the command
- updates persistent state as needed
- records side effects
- returns a new observation

### Outputs recorded during the rollout

The rollout can produce several kinds of evidence:

- raw command outputs such as stdout/stderr
- backend state changes such as task/calendar/file/config updates
- explicit effect traces such as emails sent or events created
- command history and step usage

These are later merged into the evaluator-facing normalized state.

## The V2 evaluation contract

Another team integrating V2 should not describe the evaluator loosely as “grading commands” or “checking answers.” The evaluator is stricter and more structured than that.

### Normalized evaluator state

The environment materializes a normalized state, commonly denoted `S-hat`, that includes:

- task/instruction context
- bounded command history
- recent command outputs
- merged backend state
- explicit effect traces
- configuration state where relevant

This normalized state is what the evaluator actually checks.

### Checker types

Tasks can combine several checker types:

- `state`: validate final backend state values
- `effect`: validate observable side effects
- `config`: validate relevant configuration values
- `output`: validate command outputs when needed
- optional `llm`: validate open-ended semantic criteria

In practice, the hard benchmark is primarily driven by `state`, `effect`, and `config` checks.

### Result-first semantics

The evaluator is result-first:

- it checks whether the right effects and state transitions happened
- it does not require one exact command trace
- multiple trajectories can pass if they satisfy the same evaluated outcome

This point should be preserved in any unified documentation. If another team rewrites V2 as an exact-match benchmark, that rewrite is incorrect.

### Metrics

V2 commonly reports both:

- **full-pass accuracy**
- **partial-credit average score**

These are not redundant metrics.

Full-pass accuracy requires all required checks to pass.

Average score is the dataset mean of the per-task weighted aggregate score computed from the same result-first checks. That means a task can receive substantial partial credit even when it fails to satisfy the full contract.

### Provider-aware accounting

When external LLM endpoints are used, interactive runs can be affected by retries, filtered outputs, compact fallbacks, or upstream failures. V2 therefore keeps provider-aware accounting fields such as:

- `provider_failures`
- `provider_impacted_tasks`
- `provider_adjusted_accuracy`

These fields do not redefine the evaluator; they make interactive rollout results more interpretable.

## The flagship V2 benchmark family

The current V2 flagship benchmark family is:

- `hard_decision_workflow`

This family is designed to test whether an agent can navigate difficult workflow situations where state is incomplete, stale, conflicting, or partially correct.

### Primary ability buckets

The current hard suite is organized around six primary ability buckets:

- `duplicate_avoidance`
- `gap_completion`
- `information_transfer`
- `multi_source_reasoning`
- `state_repair`
- `workflow_completion`

These buckets are compact summaries, not the full taxonomy. They are useful for reporting but do not replace the scenario-level breakdown.

### Scenario families

The official hard profile currently spans `17` scenario families, including:

- Already Done Skip
- Duplicate Avoidance
- Completion Gap
- Existing State
- Interrupted Workflow Resume
- Delivery Update
- Channel Incident Recovery
- Inbox
- Branch Resolution
- Contradictory Source Resolution
- Multi-Source Decision
- State Repair
- Wrong-State Replacement
- Release Recovery Runbook
- Release Gate
- Daily Operations Commitment Loop
- Operations Review

The full current mapping from scenario family to generator slug and primary ability is documented in [Hard Benchmark Reference](hard-benchmark.md).

### Why the hard suite is hard

The hard suite is not mainly difficult because the checker is arbitrary or brittle. It is difficult because the workflow semantics themselves are difficult:

- partial existing state must be inspected before acting
- already-correct objects should be preserved
- wrong state may need repair rather than duplication
- replacement tasks require discriminating stale objects from valid ones
- branch choice may depend on multiple sources
- success may require end-to-end closure rather than a single correct sub-action

That distinction matters when another team writes benchmark documentation. V2 should be described as **workflow-hard**, not checker-hard.

## Current official release profile

The current official released snapshot is:

- `17` hard scenarios
- `362` hard tasks
- `1616` total tasks

Default split statistics for the current release are:

- `train 967`
- `dev 321`
- `test 328`

These numbers are release statistics, not permanent benchmark limits.

They should be introduced with wording like:

> The current official release profile contains ...

and not with wording that implies the benchmark family cannot be regenerated at different scales.

## Execution modes

V2 supports four execution modes:

- `mock`
- `multi`
- `real`
- `hybrid`

The exact semantics are documented in [execution-modes.md](execution-modes.md), but for integration purposes the most important summary is:

- `multi` is the benchmark-default mode
- execution modes change backend realization, not benchmark semantics
- the same task family can be evaluated under different backend realizations

If another team writes umbrella docs, execution modes usually belong in shared runtime documentation rather than in a V2-only benchmark overview.

## History modes in LLM rollouts

The LLM rollout client currently supports three history policies:

- `full`
- `summary`
- `auto`

The current default is `full`.

This matters because it changes how much interaction history is provided to the LLM during rollout. It does **not** change the task definition or evaluator contract, but it can affect rollout behavior and should therefore be documented in evaluation-facing docs rather than in one-line benchmark summaries.

The canonical place for that is [evaluation.md](evaluation.md).

## Main code areas another team should understand

For integration or unified documentation work, these are the important code regions.

### 1. Repo-facing docs

- `README.md`
- `docs/README.md`
- `docs/quickstart.md`
- `docs/evaluation.md`
- `docs/execution-modes.md`
- `docs/openclaw-setup.md`
- `docs/task-generation.md`
- `docs/hard-benchmark.md`
- `docs/project-structure.md`

These are the primary materials another team should use when writing umbrella documentation.

### 2. Runtime and environment

- `examples/`
- `openclaw_env/backend/`
- `openclaw_env/core/`
- `openclaw_env/skills/`

This layer contains the shared runtime substrate: execution routing, backend implementations, skill families, and environment abstractions.

### 3. Benchmark generation

- `openclaw_env/tasks/`
- `scripts/generate_tasks.py`

This layer defines the generator-backed benchmark family. If another team wants to understand how release counts are formed or how scenarios are defined, this is the layer to inspect.

### 4. Task and dataset artifacts

- `openclaw_env/data/tasks/`
- `openclaw_env/data/datasets/`

This layer contains generated task specs, split files, and coverage reports.

Important generated artifacts include:

- `openclaw_env/data/datasets/generator_coverage_report.json`
- `openclaw_env/data/datasets/hard_split_coverage_report.json`
- released split files under `openclaw_env/data/datasets/`

### 5. Evaluation

- `openclaw_env/evaluation/`
- `openclaw_env/utils/episode_memory.py`
- `examples/train_and_eval.py`

This layer defines result-first checks, rollout behavior, reporting, and LLM-facing history policies.

### 6. Manuscript and figures

- `Latex/ClawArena_V2/main/`

If another team is integrating paper materials or preparing a unified paper/docs index, this is the canonical V2 manuscript path.

## How to document V2 inside a larger repo

If another team is preparing one shared repository homepage, the cleanest approach is a three-layer documentation structure.

### Layer 1: umbrella README

The root README should explain:

- what the shared environment/repo is
- that ClawArena-V2 is one benchmark line inside it
- where to find V2-specific documentation

It should not inline the full V2 benchmark taxonomy or all generation details.

### Layer 2: shared infrastructure docs

The following topics are usually better written once at the repo level:

- installation
- OpenClaw setup
- provider bootstrap
- execution modes
- common evaluation entrypoints
- project structure

These are shared runtime concerns, not unique V2 claims.

### Layer 3: V2-specific docs

The following topics should remain clearly labeled as V2-specific:

- benchmark overview
- scenario families and ability buckets
- current release profile
- generator semantics
- normalized evaluator state and result-first checks
- V2 figures, tables, and manuscript claims

This is what keeps a unified repo readable and scientifically accurate.

## Common mistakes to avoid

These are the mistakes another development team is most likely to make when summarizing V2 quickly.

### Mistake 1: describing V2 as a static benchmark

Wrong idea:

- “V2 is a set of fixed benchmark questions.”

Correct idea:

- V2 is a generator-backed benchmark family with a current official release profile.

### Mistake 2: flattening V2 into “tool use”

Wrong idea:

- “V2 measures whether an agent can call tools.”

Correct idea:

- V2 measures whether an agent can execute multi-step workflows correctly under evolving state.

### Mistake 3: describing evaluation as exact-match

Wrong idea:

- “The agent must match the reference trajectory.”

Correct idea:

- V2 is result-first and allows multiple valid trajectories to pass if they yield the same evaluated outcome.

### Mistake 4: presenting release counts as permanent limits

Wrong idea:

- “V2 consists of exactly 362 hard tasks.”

Correct idea:

- The current official hard snapshot contains 362 hard tasks, but counts are generator-configurable.

### Mistake 5: mixing shared setup with benchmark-specific claims

Wrong idea:

- mixing provider setup, benchmark taxonomy, and paper results on one page

Correct idea:

- keep setup docs shared; keep V2 benchmark semantics and claims under V2-specific docs

### Mistake 6: omitting the operational path

Wrong idea:

- assuming another team only needs high-level benchmark prose

Correct idea:

- include the editable install path, generation command, evaluation entrypoint, execution-mode requirements, and the location of canonical V2 docs

## Suggested wording blocks for external teams

### Very short form

- **ClawArena-V2**: interactive benchmark for command-line agents
- **Core contract**: command-by-command execution under persistent state
- **Evaluation**: normalized state plus result-first checks
- **Generation**: scenario-family templates with configurable release counts
- **Flagship suite**: `hard_decision_workflow`

### Medium form

> ClawArena-V2 is a benchmark line for evaluating command-line agents in stateful execution environments. Agents interact step by step, environment state evolves after each action, and success is determined by result-first checks over normalized state and observable side effects. The current flagship suite, `hard_decision_workflow`, is generator-backed and released through configurable scenario profiles rather than one permanently fixed task list.

## Practical checklist for external developers

Use this checklist before integrating V2 into a broader repo or rewriting docs around it.

1. Read [README](../README.md), [Hard Benchmark Reference](hard-benchmark.md), and [Evaluation Guide](evaluation.md).
2. Keep V2 benchmark docs under an explicitly labeled V2 path.
3. Treat current counts as release statistics, not immutable limits.
4. Keep setup and execution-mode docs shared if the larger repo uses the same runtime.
5. Keep scenario taxonomy, release profiles, and evaluator semantics V2-specific.
6. Point paper/manuscript references to `Latex/ClawArena_V2/main/`.
7. If summarizing the benchmark briefly, emphasize interactive execution, generator-backed tasks, and normalized result-first evaluation.
8. If another team will actually run the code, also point them to:
   - [quickstart.md](quickstart.md)
   - [evaluation.md](evaluation.md)
   - [openclaw-setup.md](openclaw-setup.md)
   - [task-generation.md](task-generation.md)

## Recommended next documents for a larger repo

If another team is building a unified documentation tree around this codebase, the most useful next documents are:

- one umbrella README
- one V2 overview page
- one shared setup guide
- one shared execution-modes guide
- one shared project-structure guide
- one shared evaluation entrypoint guide

This is enough to preserve V2 semantics without overloading the root README.

## Final recommendation

If you need one sentence that reliably explains ClawArena-V2 to another development team, use this:

**ClawArena-V2 is an interactive execution benchmark line built on a shared OpenClaw-style environment, with generator-backed task creation and normalized result-first evaluation over persistent state.**

That sentence is short enough for an umbrella repo and still accurate enough to avoid the most common integration mistakes.
