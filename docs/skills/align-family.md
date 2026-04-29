# `align-family`

> Drive the full migration for an op family — audit, delegate per-op
> alignment to `align-op`, run cross-op cleanup. Two terminal outcomes:
> **SUCCESS** opens a PR; **CLEANUP_REGRESSION** exits blocked without a
> PR when post-cleanup tests fail.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* You're migrating an entire op family (e.g. all reductions, all
  norms) to a new manifest layout in one pass.
- *(TODO)* You want a single orchestrator that handles audit → per-op
  alignment → cross-op cleanup → PR, instead of running each step by hand.

## Inputs

- *(TODO)* Family name.
- *(TODO)* Manifest spec to align against.

## Outputs

- *(TODO)* On SUCCESS — a PR with the full family migration.
- *(TODO)* On CLEANUP_REGRESSION — a blocked-state report; no PR.

## Example

*(TODO — to be filled in.)*

## Implementation notes

*(TODO — composition with `audit-family`, `align-op`; failure recovery.)*

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/align-family/SKILL.md"
%}
