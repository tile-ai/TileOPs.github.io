# `implement-op`

> Modify op code to match the manifest-declared interface, making spec
> tests pass.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* The op already exists and the gap to its manifest is small
  (minor-delta path of [`align-op`](align-op.md)).
- *(TODO)* You have failing spec tests written by [`test-op`](test-op.md)
  and want them green.

## Inputs

- *(TODO)* `op_name`.
- *(TODO)* Failing spec tests as the implementation target.

## Outputs

- *(TODO)* Updated op source file; spec tests pass.

## Example

*(TODO — to be filled in.)*

## Implementation notes

*(TODO — boundary with `scaffold-op` (greenfield) and `align-op`
(orchestrator).)*

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/implement-op/SKILL.md"
%}
