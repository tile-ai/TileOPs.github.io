# `scaffold-op`

> Scaffold a new T2 (L1-direct) Op file from a single
> `tileops/manifest/` entry by following the 7-step playbook in
> [`docs/design/ops-design.md`](../design/ops-design.md). Emits the 17
> scaffold slots (S1–S7, S12–S21); leaves family-specific protocol
> variables, optional hooks, and kernel implementations to downstream
> skills.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* Greenfield: a manifest entry exists but no op source file
  yet.
- *(TODO)* Interface-redesign path of [`align-op`](align-op.md), after
  archiving the prior implementation.

## Inputs

- *(TODO)* Manifest entry (the single source of truth for slot values).

## Outputs

- *(TODO)* A new op source file with the 17 scaffold slots filled.

## Example

*(TODO — to be filled in.)*

## Implementation notes

- *(TODO)* Slot rules are documented in
  [Op Interface Reference](../design/ops-design-reference.md).
- *(TODO)* Downstream skills fill family-specific protocol variables,
  optional hooks, and kernel bodies.

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/scaffold-op/SKILL.md"
%}
