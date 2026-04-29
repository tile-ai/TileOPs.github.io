# `bench-op`

> Fix the benchmark file to work with the new Op interface. Run the
> benchmark, fix errors, repeat until it produces numbers.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* An op's interface changed and its existing benchmark file
  no longer runs.
- *(TODO)* You want benchmark numbers as part of the per-op alignment
  pipeline (run by [`align-op`](align-op.md) after `test-op` passes).

## Inputs

- *(TODO)* `op_name`.

## Outputs

- *(TODO)* A working `benchmarks/<op>.py` (or equivalent) plus its
  generated `profile_run.log`.

## Example

*(TODO — to be filled in.)*

## Implementation notes

*(TODO — relationship to `tests/` vs `benchmarks/` separation; profiler
output expectations.)*

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/bench-op/SKILL.md"
%}
