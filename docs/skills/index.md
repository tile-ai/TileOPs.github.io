# TileOPs Skills

TileOPs ships a set of **Claude Code skills** — small, focused capabilities
an AI agent (or a human via Claude Code) can invoke to perform
project-specific tasks: scaffolding a new operator, aligning one to the
manifest, running benchmarks, and so on.

## Two audiences, two views

Each skill is documented twice, for different readers:

| Layer | Where | Reader | Style |
|:--|:--|:--|:--|
| Skill source (`SKILL.md`) | [`.claude/skills/`](https://github.com/tile-ai/TileOPs/tree/main/.claude/skills) in `tile-ai/TileOPs` | The Claude Code agent at runtime | Terse, instruction-dense |
| Skill guide (this site) | `docs/skills/<name>.md` | Human contributors | When to use, inputs/outputs, examples, gotchas |

The skill source is the source of truth. Each guide page on this site
embeds its skill's `SKILL.md` at the bottom, so the two views never drift.

## Using skills

Skills live in `tile-ai/TileOPs` under `.claude/skills/`. Claude Code picks
them up automatically when run inside a clone of that repository — no
separate install.

## At a glance

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="## At a glance"
   end="## Skills in detail"
%}

## Trust model  ·  who may write what

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="## Trust model"
   end="## Maintenance"
%}

For per-skill detail and the embedded agent contracts, see the
[Catalog](catalog.md) and individual skill pages in the navigation.
