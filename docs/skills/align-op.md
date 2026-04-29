# `align-op`

> Per-op orchestrator that brings a single op into alignment with its
> manifest entry. Classifies the op (green field / interface redesign /
> minor delta), dispatches to the right path, then runs the shared
> downstream: test → bench → validate → flip status → report.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* You know the specific op you want to bring into alignment,
  and don't need a family-wide sweep.
- *(TODO)* Family-scoped equivalent: see [`align-family`](align-family.md).

## Inputs

- *(TODO)* `op_name`.
- *(TODO)* Manifest entry serving as the alignment target.

## Outputs

- *(TODO)* Updated op code matching the manifest contract.
- *(TODO)* Test + bench evidence; status flipped on success.

## Example

*(TODO — to be filled in.)*

## Implementation notes

- *(TODO)* Three dispatch paths:
    - **Green field** — delegates to [`scaffold-op`](scaffold-op.md).
    - **Interface redesign** — archive existing → rescaffold → port.
    - **Minor delta** — delegates to [`implement-op`](implement-op.md).

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/align-op/SKILL.md"
%}
