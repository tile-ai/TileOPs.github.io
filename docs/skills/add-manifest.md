# `add-manifest`

> Generate or re-align one `tileops/manifest/` entry from a reference-API
> docs URL. Caller provides the manifest key (`op_name`); the skill writes
> that one entry. Idempotent.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* You have a new op with a stable upstream reference (PyTorch
  / Triton / library doc URL) and want a manifest entry generated from
  it.
- *(TODO)* You need to re-derive an existing entry's reference-derivable
  fields after the upstream API moved.

## Inputs

- *(TODO)* `op_name` — manifest key.
- *(TODO)* Reference-API documentation URL.

## Outputs

- *(TODO)* One `tileops/manifest/<op_name>.json` (or equivalent) entry,
  written or updated in place.

## Example

*(TODO — to be filled in.)*

## Implementation notes

*(TODO — relationship to `fix-manifest`, idempotency guarantees.)*

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/add-manifest/SKILL.md"
%}
