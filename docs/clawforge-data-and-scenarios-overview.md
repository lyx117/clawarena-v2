# ClawForge Data and Scenario Overview

Back to [README](../README.md) · See also [hard-benchmark.md](hard-benchmark.md), [task-generation.md](task-generation.md), [evaluation.md](evaluation.md), and [clawforge-integration-guide.md](clawforge-integration-guide.md)

## What ClawForge is

ClawForge is an interactive benchmark for command-line agents. The key idea is simple: the agent does not just answer once in text. It has to look at the current state, issue commands step by step, and actually finish the workflow in the environment.

That means the benchmark is not asking, “Can the model describe the right plan?” It is asking, “Can the agent inspect what already exists, take the right action, and leave the environment in the right final state?”

## How the data is produced

The ClawForge data is not one fixed handwritten list of benchmark questions. It is generated from reusable scenario families.

A scenario family is a pattern like:

- something is already done, so the agent should not duplicate it
- part of the workflow is missing, so the agent should only fill the gap
- the current state is wrong or stale, so the agent should repair or replace it
- several sources disagree, so the agent has to choose the right branch

For each generated task, the pipeline combines a few pieces:

- a scenario template
- grounded variables and filled slots
- an initial environment state
- an expected workflow structure
- an executable validator
- metadata used for reporting and grouping

So a generated task is more than an instruction string. It also includes the environment setup, the benchmark bookkeeping, and the checks that define success.

This is why current counts such as `17` hard scenarios, `362` hard tasks, and `1616` total tasks should be read as **official release stats for the current snapshot**, not as hard limits on what ClawForge can contain.

## How the scenarios are produced

Each scenario family is built to capture one recurring workflow pattern or failure mode. The generator then creates many concrete tasks from that family by varying the starting state and task details.

In plain terms, the generator keeps the same core situation but changes the concrete instance. One task might involve one inbox thread, one calendar item, and one deadline. Another task from the same family keeps the same pattern but changes the names, timing, missing steps, or conflicting evidence.

That is what makes the benchmark scalable. The benchmark is not manually rewritten from scratch each time. The generator reuses the same family semantics and instantiates many concrete rollouts.

Here are a few examples.

### Already done / duplicate avoidance

This kind of scenario is about checking whether the required work already exists.

The agent should look first, realize the task is already complete or already represented in the environment, and avoid creating duplicates. These scenarios are useful because many agents are too eager to act and create extra objects even when the right answer is to leave correct state alone.

### Partial state / gap completion

This kind of scenario is about finding the missing piece of a workflow.

Maybe a task exists but the calendar event does not. Maybe the draft file exists but the follow-up action is missing. The agent should add only what is missing instead of rebuilding the whole workflow.

### Stale or wrong state / repair or replacement

This kind of scenario is about recognizing that the current state is not just incomplete but actually wrong.

Sometimes the right move is to repair an existing object. Sometimes the right move is to replace it with a correct one. These scenarios are hard because the agent has to distinguish “missing,” “already correct,” and “actively wrong.”

### Conflicting evidence / branch choice

This kind of scenario is about combining multiple signals before acting.

For example, an email may say one thing while a calendar entry or another source suggests something else. The agent has to resolve the conflict and choose the right branch instead of blindly following the first piece of evidence it sees.

### Workflow closure across tasks, calendar, email, and files

This kind of scenario is about finishing the whole workflow rather than getting one local step right.

An agent may correctly update a task but forget the follow-up file, calendar event, or communication step. These scenarios test whether the agent can carry a workflow all the way to closure across multiple surfaces.

## What makes the benchmark hard

The benchmark is hard because the workflow situations are hard, not because the checker is trying to trick the model.

The hard parts are things real agents struggle with:

- part of the correct state already exists
- the agent must avoid duplicating valid work
- the next step depends on choosing the right branch
- wrong state has to be repaired instead of ignored
- stale state has to be replaced without breaking good state
- success means finishing the entire workflow, not just one sub-step

That is why ClawForge is more informative than a one-shot benchmark. It exposes the difference between an agent that can talk about the workflow and an agent that can actually complete it.

## How evaluation works in simple terms

After a rollout, the benchmark does not just look at the final text the agent produced. It builds a normalized evaluation state that summarizes what happened in a consistent way.

In simple terms, that normalized state includes things like:

- the recent command history
- command outputs
- the current backend state
- effect traces such as created tasks, updated events, sent emails, or changed files

The evaluator then checks what changed in the environment and what observable effects actually happened.

This is why multiple trajectories can still pass. If two agents take different command sequences but end in the same correct evaluated outcome, they can both succeed. The benchmark is result-first, not exact-trajectory-first.

The benchmark also reports two different metrics:

- **full-pass accuracy**, which asks whether the task fully satisfied the required checks
- **average score**, which gives partial credit when an agent makes meaningful progress but does not complete the entire task

That distinction matters because an agent can do a lot of the workflow correctly and still miss the final closure step.

## Current official profile

The current official ClawForge hard profile is:

- `17` hard scenarios
- `362` hard tasks
- `1616` total tasks

These numbers describe the current official release snapshot. They are not hard upper bounds on the benchmark family.

The current flagship benchmark family is:

- `hard_decision_workflow`

The six main reporting buckets in the current hard suite are:

- `duplicate_avoidance`
- `gap_completion`
- `information_transfer`
- `multi_source_reasoning`
- `state_repair`
- `workflow_completion`

## Where to look next

If you want the formal benchmark taxonomy, use [Hard Benchmark Reference](hard-benchmark.md).

If you want the generator knobs and output locations, use [Task Generation Guide](task-generation.md).

If you want rollout settings, metrics, and output reports, use [Evaluation Guide](evaluation.md).

If you want the more technical repo-facing integration explanation, use [ClawForge Integration Guide](clawforge-integration-guide.md).
