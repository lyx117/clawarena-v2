# Results Snapshot

Back to [README](../README.md) · See also [hard-benchmark.md](hard-benchmark.md) and [evaluation.md](evaluation.md)

## What this page is

This page records a repository-facing snapshot of the current benchmark results that are also used in the manuscript. It is meant to help readers understand the benchmark profile and the kind of separation the benchmark exposes across models.

This is **not** a live leaderboard. These figures summarize one documented release snapshot and should be read together with the benchmark configuration, execution mode, and provider caveats described in the rest of the docs.

## Release-snapshot framing

The current result snapshot should be read with three constraints in mind:

- it reflects one official hard-benchmark release profile rather than all possible generator outputs
- it reflects one evaluated run set rather than a continuously refreshed scoreboard
- provider-aware fields such as retries, filtered outputs, or fallback behavior can affect interactive runs even when the benchmark definition itself is unchanged

For the benchmark design and scenario inventory behind these results, see [hard-benchmark.md](hard-benchmark.md). For the evaluation protocol and reporting fields, see [evaluation.md](evaluation.md).

## Main results view

<p align="center">
  <img src="assets/main-results-scatter.png" alt="Main release-snapshot results scatter for ClawForge" width="760">
</p>

The scatter view is useful because it shows two signals at once:

- strict outcome quality through full-pass accuracy
- near-miss behavior through average partial-credit score

That distinction matters in ClawForge because a rollout can make substantial progress on a stateful workflow and still fail strict completion.

## Benchmark composition view

<p align="center">
  <img src="assets/hard-benchmark-composition.png" alt="Composition of the current ClawForge hard benchmark release snapshot" width="680">
</p>

This figure is included here because result interpretation depends on benchmark composition. A model that looks strong on aggregate may still be weak on repair-oriented or branch-resolution-heavy slices.

## How to cite these figures in repo docs

When using these figures in documentation or downstream summaries, prefer wording like:

> The repository includes a documented release snapshot of benchmark results for the current hard profile.

Avoid wording like:

> These are the permanently current benchmark rankings.

The second claim is too strong for a generator-backed benchmark whose release counts and evaluated model set can evolve over time.
