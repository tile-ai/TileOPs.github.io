# `test-op`

> Write tests for the target spec using PyTorch as ground truth, verify
> they fail on the current code.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* Spec-first development: you have a manifest entry and want
  failing tests written before implementation.
- *(TODO)* Called by [`align-op`](align-op.md) before
  [`implement-op`](implement-op.md) in the alignment pipeline.

## Inputs

- *(TODO)* `op_name`.
- *(TODO)* Manifest entry serving as the spec.

## Outputs

- *(TODO)* New / updated `tests/<op>.py` with cases derived from the
  manifest, verified to fail against the current code.

## Example

*(TODO — to be filled in.)*

## Implementation notes

*(TODO — PyTorch-as-ground-truth pattern; verifying tests fail before
handing off to `implement-op`.)*

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/test-op/SKILL.md"
%}
