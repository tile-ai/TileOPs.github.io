# `audit-family`

> Compare each op's code signature against its manifest spec, classify
> gaps, produce a structured report.

!!! info "Page under construction"
    The human-facing guide below is a placeholder. The authoritative
    agent-facing contract is embedded in the **Reference** section at
    the bottom of this page.

## When to use

- *(TODO)* You want a read-only diagnostic across an op family before
  deciding whether to run [`align-family`](align-family.md).
- *(TODO)* You need a structured artifact (gaps, classifications) to
  share with humans or feed into a follow-up skill.

## Inputs

- *(TODO)* Family name.

## Outputs

- *(TODO)* A structured report listing each op's classification (green
  field / interface redesign / minor delta / aligned).

## Example

*(TODO — to be filled in.)*

## Implementation notes

*(TODO — pure read-only; no code changes; called by `align-family` as
the first step.)*

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/audit-family/SKILL.md"
%}
