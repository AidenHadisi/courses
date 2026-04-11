# Claude Code Skills

A **skill** is a reusable instruction set that teaches Claude how to handle a specific task consistently. Skills are defined as directories containing a `SKILL.md` file and can be shared across projects.

---

## Structure

Every skill is a directory with a `SKILL.md` file. The file has two sections separated by frontmatter dashes:

```
~/.claude/skills/pr-description/
├── SKILL.md
├── scripts/      # executable code
├── references/   # additional documentation
└── assets/       # images, templates, other data
```

```markdown
---
name: pr-description
description: Writes pull request descriptions. Use when creating a PR, writing a PR,
             or when the user asks to summarize changes for a pull request.
---

When writing a PR description:

1. Run `git diff main...HEAD` to see all changes on this branch
2. Write a description following this format:
...
```

### Metadata Fields

| Field | Required | Constraints | Purpose |
|---|---|---|---|
| `name` | Yes | Lowercase, numbers, hyphens; max 64 chars | Unique identifier; should match directory name |
| `description` | Yes | Max 1,024 chars | **Matching criteria** — what Claude reads to decide whether to activate |
| `allowed-tools` | No | Comma-separated tool names | Restricts which tools Claude may use when the skill is active |
| `model` | No | Model identifier | Specifies which Claude model to use |

---

## How Skill Loading Works

Claude Code loads skills **lazily** to keep startup fast and context clean:

1. **At startup** — scans skill directories, loads only `name` and `description`
2. **On request** — compares the user's message against all descriptions using semantic matching
3. **On match** — prompts the user to confirm before loading full skill content
4. **After confirmation** — reads the complete `SKILL.md` and follows its instructions

The full skill body never enters context until needed, and the user is always aware of what's being loaded.

---

## Writing Effective Descriptions

The description is the single most important field. Claude uses it exclusively for matching — vague descriptions produce unreliable activation.

A good description answers two questions:
1. **What does the skill do?**
2. **When should Claude use it?**

If a skill isn't triggering when expected, add more keywords that match how you actually phrase requests. The language in the description should mirror the language in your prompts.

> [!IMPORTANT] Don't write "helps with docs." Write "Rewrites API documentation. Use when the user asks to update, reformat, or improve documentation files."

---

## Tool Restrictions with `allowed-tools`

`allowed-tools` restricts which tools Claude can use without asking permission while the skill is active. Useful for read-only workflows, security-sensitive tasks, or enforcing guardrails.

```markdown
---
name: codebase-onboarding
description: Helps new developers understand how the system works.
allowed-tools: Read, Grep, Glob, Bash
model: sonnet
---
```

With this configuration, Claude can explore the codebase but cannot edit or write files during this skill's execution.

If `allowed-tools` is omitted, Claude uses its normal permission model — no restrictions added.

---

## Progressive Disclosure

Skills share Claude's context window with the conversation. Loading a 2,000-line `SKILL.md` every time a skill activates wastes context and is hard to maintain.

**The pattern:** keep essential instructions in `SKILL.md` and put reference material in supporting files that Claude reads only when the task calls for it.

```markdown
# Architecture Skill

For general questions, answer from your training knowledge.

For system design questions, read `references/architecture-guide.md` first.
For component placement questions, read `references/component-map.md` first.
```

Claude loads `architecture-guide.md` only when someone asks about system design — other queries never touch that file. The context window holds a table of contents, not the whole book.

**Rule of thumb:** keep `SKILL.md` under 500 lines. If you're exceeding that, split content into reference files.

---

## Scripts

Scripts in `scripts/` run without loading their source into context — only the output consumes tokens.

In `SKILL.md`, tell Claude to **run** the script, not read it:

```markdown
Before starting, run `scripts/validate-env.sh` to verify the environment is configured correctly.
```

Good uses for scripts:
- Environment validation
- Consistent data transformations
- Operations that are more reliable as tested code than generated code

---

## Priority Hierarchy

When two skills share the same name, Claude applies this order (highest wins):

| Priority | Location | Use case |
|---|---|---|
| **1. Enterprise** | Managed org settings | Enforce company-wide standards |
| **2. Personal** | `~/.claude/skills/` | Individual preferences, works across all projects |
| **3. Project** | `.claude/skills/` in repo | Project-specific workflows, checked into the repo |
| **4. Plugins** | Installed plugins | Third-party skill packages |

> [!TIP] Use specific names (`frontend-review`, `backend-review`) over generic ones (`review`) to avoid accidental conflicts.

---

## Creating a Personal Skill

```bash
mkdir -p ~/.claude/skills/my-skill
# create ~/.claude/skills/my-skill/SKILL.md
```

Restart Claude Code after any changes — skill discovery happens at startup.

## Lifecycle

| Action | How |
|---|---|
| Create | Add a directory with `SKILL.md` inside a skills folder |
| Update | Edit `SKILL.md` or supporting files |
| Remove | Delete the skill's directory |
| Activate changes | Restart Claude Code |

---

## Sharing and Distribution

| Method | Location | Who gets it | Best for |
|---|---|---|---|
| **Repository commit** | `.claude/skills/` | Anyone who clones the repo | Team standards, project-specific workflows |
| **Plugin** | Plugin marketplace | Any Claude Code user who installs it | General-purpose skills useful beyond one team |
| **Enterprise managed settings** | Org admin config | Entire organization, highest priority | Mandatory standards, compliance, security requirements |

### Repository Commit

The simplest method. Commit skills to `.claude/skills/` and they're shared automatically through normal Git workflows — clone, push, pull. The `.claude/` directory holds agents, hooks, skills, and settings together.

### Plugins

Create a `skills/` directory inside your plugin project with the same structure as `.claude/skills/` (each skill gets its own folder with a `SKILL.md`). Publish to a marketplace; other users install it into Claude Code.

Best when skills are general enough to be useful beyond your immediate team.

### Enterprise Managed Settings

Administrators deploy skills organization-wide. Enterprise skills have the highest priority — they override personal, project, and plugin skills with the same name.

The managed settings file can also restrict which plugin sources are allowed:

```json
"strictKnownMarketplaces": [
  { "source": "github", "repo": "acme-corp/approved-plugins" },
  { "source": "npm", "package": "@acme-corp/compliance-plugins" }
]
```

Use enterprise deployment for anything that *must* be consistent — not just *should* be.

---

## Skills and Subagents

Subagents don't inherit skills automatically. When a task is delegated to a subagent, it starts with a clean context.

Two important constraints:

- **Built-in agents** (Explorer, Plan, Verify) — cannot access skills at all
- **Custom subagents** defined in `.claude/agents/` — can use skills, but only when explicitly listed in frontmatter

Skills listed in a custom agent are loaded when the subagent starts, not matched on demand like in the main conversation.

### Custom Agent With Skills

Create an agent file in `.claude/agents/` (or use `/agents` in Claude Code to generate one):

```markdown
---
name: frontend-security-accessibility-reviewer
description: "Use this agent when you need to review frontend code for accessibility..."
tools: Bash, Glob, Grep, Read, WebFetch, WebSearch, Skill
model: sonnet
color: blue
skills: accessibility-audit, performance-check
---
```

The `skills` field lists skill names to load. The skills must exist in `.claude/skills/`.

This pattern is useful when:
- Different subagents need different expertise (frontend reviewer vs. backend reviewer)
- You want isolated task delegation with specific knowledge loaded
- You need to enforce standards in delegated work without relying on prompts

---

## Troubleshooting

Most skill failures fall into a few predictable categories. Start with the validator before debugging anything else.

```bash
# Install via uv, then run from your skill directory or pass the path
uv tool install agent-skills-validator
```

The validator catches structural problems (bad YAML, wrong file names, missing required fields) before you spend time chasing other causes.

### Skill Doesn't Trigger

Cause: the description doesn't overlap enough with how you're phrasing requests.

- Compare your description to the exact words you're using
- Add trigger phrases users would actually say ("help me profile this", "why is this slow", "make this faster")
- If a variation fails to trigger, add those keywords to the description

### Skill Doesn't Load

Check these structural requirements:

- `SKILL.md` must be inside a named directory, not at the skills root
- File name must be exactly `SKILL.md` — all caps on "SKILL", lowercase "md"

```bash
claude --debug   # look for loading errors mentioning your skill name
```

### Wrong Skill Gets Used

Cause: two descriptions are too similar and Claude picks the wrong one.

Make descriptions more distinct and specific. Specificity also prevents conflicts with other similar-sounding skills.

### Priority Conflicts

If your personal skill is being ignored, a higher-priority skill (enterprise or project) likely shares the same name. Options:
- Rename your skill to something more distinct (usually the easier path)
- Talk to your admin about the enterprise skill

### Plugin Skills Not Appearing

Clear the cache, restart Claude Code, and reinstall the plugin. If skills still don't appear, the plugin structure is likely wrong — run the validator on the plugin's skill directory.

### Runtime Errors

The skill loads but fails during execution. Common causes:

| Symptom | Fix |
|---|---|
| Missing dependency | Install the package; document requirements in the skill description |
| Script won't run | `chmod +x` on any scripts the skill references |
| Wrong path on Windows | Use forward slashes everywhere, including on Windows |

### Quick Checklist

- **Not triggering** → improve description, add trigger phrases
- **Not loading** → check directory structure, file name casing, YAML syntax
- **Wrong skill used** → make descriptions more distinct
- **Being shadowed** → check priority hierarchy, rename if needed
- **Plugin skills missing** → clear cache, reinstall, validate plugin structure
- **Runtime failure** → check dependencies, `chmod +x`, forward slashes in paths

---

## Skills vs Other Customization Options

Claude Code has five customization mechanisms. Each solves a different problem — knowing which to reach for prevents unnecessary complexity.

### Quick Reference

| Feature | Activation | Scope | Best for |
|---|---|---|---|
| **CLAUDE.md** | Every conversation, always | Project-wide | Always-on standards and constraints |
| **Skills** | On demand, request-matched | Current conversation | Task-specific expertise |
| **Subagents** | Explicitly dispatched | Isolated context | Delegated work with separate tool access |
| **Hooks** | Event-driven (file save, tool call) | Automated side effects | Validation, linting, automated operations |
| **MCP servers** | Persistent connection | External tools | Integrations, databases, external APIs |

### CLAUDE.md vs Skills

**CLAUDE.md** loads into every conversation. Use it for things that always apply regardless of what you're doing.

**Skills** load only when a request matches their description. Use them for knowledge that's only relevant sometimes — loading a PR review checklist while writing new code just wastes context.

| Use CLAUDE.md for | Use Skills for |
|---|---|
| Project-wide coding standards | Task-specific procedures |
| Hard constraints ("never modify the schema") | Expertise only relevant sometimes |
| Framework and style preferences | Detailed checklists that would clutter every conversation |

> [!TIP] Audit your CLAUDE.md periodically. If something in it only matters during certain tasks (e.g. "when writing tests, do X"), move it to a skill.

### Skills vs Subagents

**Skills** add knowledge to your *current* conversation. Instructions join the existing context and inform how Claude reasons about the current task.

**Subagents** run in a *separate* context. They receive a task, work independently, and return results — isolated from the main conversation.

Use subagents when you want to delegate a task with different tool access or when isolation between that work and your main context is important.

### Skills vs Hooks

**Hooks** are event-driven — they fire when something happens (a file is saved, a tool is called). Use them for automated side effects: running a linter on every save, validating input before tool calls.

**Skills** are request-driven — they activate based on what you're asking. Use them for knowledge that shapes how Claude reasons, not for automated operations.

### Combining All Five

A complete setup typically uses all of them together:

- **CLAUDE.md** — always-on project standards (TypeScript strict mode, naming conventions)
- **Skills** — task-specific expertise (PR descriptions, test writing, onboarding guides)
- **Hooks** — automated operations (lint on save, validate before destructive tool calls)
- **Subagents** — isolated execution for delegated work
- **MCP servers** — external tools and integrations (databases, APIs, services)
