# `fix-manifest`

> Patch one missing structural field (`kernel_map`, `static_dims`) on an
> existing `tileops/manifest/` entry. Auto-detects the field via the
> validator or takes `--field=<name>`. Reference-derivable fields
> (`signature.*`, `shape_rules`, `dtype_combos`, `roofline`) belong to
> [`add-manifest`](add-manifest.md), not here.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* The manifest validator flagged one structural field as
  missing/invalid on an otherwise-aligned entry.
- *(TODO)* **Don't use** for reference-derivable fields — use
  [`add-manifest`](add-manifest.md).

## Inputs

- *(TODO)* `op_name`, optional `--field=<name>`.

## Outputs

- *(TODO)* Patched manifest entry; validator passes.

## Example

*(TODO — to be filled in.)*

## Implementation notes

*(TODO — split of responsibility with `add-manifest`; auto-detection
mechanism.)*

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/fix-manifest/SKILL.md"
%}
