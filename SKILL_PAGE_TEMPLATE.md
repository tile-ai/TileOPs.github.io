<!--
Template for authoring per-skill guide pages under docs/skills/.
Copy this to docs/skills/<skill-name>.md, fill in the human-facing
sections, and add the page to mkdocs.yml's `nav` under Skills.
The Reference section pulls the agent-facing SKILL.md verbatim from
the upstream tile-ai/TileOPs repo.
-->

# <Skill name>

> One-sentence description of what this skill does.

## When to use

- Use when …
- Don't use when …

## Inputs

- Required arguments / context / preconditions

## Outputs

- Artifacts produced / side effects / exit conditions

## Example

End-to-end example, including the invocation and expected output.

## Implementation notes

- Which TileOPs modules the skill touches
- Design trade-offs worth knowing

## Reference

The agent-facing source for this skill, embedded verbatim from
`tile-ai/TileOPs`:

{%
   include-markdown "../../TileOPs/.claude/skills/<skill-name>/SKILL.md"
%}
