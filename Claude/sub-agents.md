# Subagents

Subagents are specialized assistants that Claude Code can delegate tasks to. Each one runs in its own isolated conversation context, does its work, and returns only a summary to the main thread. All intermediate steps — file reads, searches, tool calls — stay contained and never pollute the main context.

## Why Subagents Matter

Every tool call, file read, and search result in the main conversation consumes space in the context window. Once that fills up, Claude starts losing track of earlier parts of the conversation.

Subagents solve this by spinning up a **separate context window**. Each subagent receives:

1. A **custom system prompt** (from your config) defining its role and behavior
2. A **task description** written by the parent agent based on your request

The subagent works independently, then returns a concise summary. The entire subagent conversation is then discarded — your main context only records the question and the answer.

> [!NOTE] The tradeoff is reduced visibility: you get the result without seeing the intermediate reasoning or tool calls the subagent made.

## Example

You want to know which service handles refunds in an unfamiliar codebase.

- **Without subagent:** Claude reads 15 files, runs several searches, traces function calls — all of it lands in your context window.
- **With subagent:** The Explore subagent does all that digging in isolation and hands back one focused answer. Your main context only records the question and the summary.

## Built-in Subagents

Claude Code ships with several subagents out of the box:

| Subagent | Purpose |
|---|---|
| General purpose | Multi-step tasks requiring both exploration and action |
| Explore | Fast searching and navigation of codebases |
| Plan | Research and codebase analysis during plan mode |

## Custom Subagents

You can define your own subagents with custom system prompts and tool access — a code reviewer, test writer, documentation generator, or anything else your workflow needs.

### Creating a Subagent

The easiest path is the `/agents` slash command, which opens the subagent management UI. From there, select **Create new agent**.

**Scope** — choose where the subagent lives:

| Scope | Availability |
|---|---|
| Project-level | Current project only |
| User-level | All projects on your machine |

You can write the config manually, but the recommended approach is to describe what you want and let Claude generate the name, description, and system prompt for you.

### Tool Access

During creation you choose which tool categories the subagent can use:

- Read-only tools
- Edit tools
- Execution tools
- MCP tools
- Other tools

Match tools to the task. A code reviewer probably doesn't need edit tools — read and analyze, not modify. Keeping execution tools enabled can still be useful for identifying pending changes.

### Model Selection

| Model | Best for |
|---|---|
| Haiku | Fast, lightweight tasks |
| Sonnet | Balanced speed and depth |
| Opus | Complex analysis |
| Inherit | Uses whatever the main conversation is running |

You also pick a **color** that appears in the UI to identify which subagent is active — useful when multiple subagents are running.

### The Config File

Subagent configs are saved as markdown files at `.claude/agents/<agent-name>.md`. Example:

```markdown
---
name: code-quality-reviewer
description: Use this agent when you need to review recently written or modified code for quality, security, and best practice compliance.
tools: Bash, Glob, Grep, Read, WebFetch, WebSearch
model: sonnet
color: purple
---

You are an expert code reviewer specializing in quality assurance, security best practices, and
adherence to project standards. Your role is to thoroughly examine recently written or modified code
and identify issues that could impact reliability, security, maintainability, or performance.
```

**Frontmatter fields:**

| Field | Purpose |
|---|---|
| `name` | Unique identifier; also used for `@agent name` mentions |
| `description` | Controls when Claude delegates to this subagent (must be a single line) |
| `tools` | Comma-separated list of allowed tools |
| `model` | `sonnet`, `opus`, `haiku`, or `inherit` |
| `color` | UI color for identification |

**System prompt** — everything below the frontmatter. This is the most important part: be specific about what the subagent should focus on and how it should structure its output. A vague system prompt produces a vague subagent.

### Automatic Delegation

By default, you have to explicitly ask Claude to use a subagent. To make Claude delegate automatically:

- Add the word **"proactively"** to the `description` field
- Include **example conversations** in the description to illustrate specific trigger scenarios

> [!TIP] The more concrete your examples in the description, the better Claude gets at knowing when to delegate without being asked.

### Invoking a Subagent

You can reference a subagent directly in your message:

```
@agent code-quality-reviewer please review my changes
```

Or just describe what you need and Claude will pick the right subagent based on the descriptions.

### Testing and Tuning

After creating a subagent, make some code changes and ask Claude to review them. If the subagent isn't being triggered when you expect, revisit the `description` field — add more specific examples and trigger scenarios.

---

## Effective Subagent Patterns

A poorly configured subagent will wander, run too long, or return output the main agent can't use. Four things fix this: good descriptions, a defined output format, obstacle reporting, and minimal tool access.

### How Config Data Gets Used

Every available subagent's `name` and `description` are injected into the main agent's system prompt. This is how the main agent decides *which* subagent to launch and *when*. The description also shapes the input prompt the main agent writes to kick off the task — it's not just a trigger condition, it's also authoring guidance.

*Example:* A generic description might cause the main agent to write "use git diff to find current changes." A description that says "you must tell the agent precisely which files to review" causes the main agent to write a specific prompt listing actual files. Same subagent, much better task handoff.

### Writing Descriptions That Shape Input Prompts

Think of the description as two-in-one:

1. **Trigger condition** — when should the main agent delegate here?
2. **Authoring hint** — what should the input prompt include?

Adding instructions like "return sources that can be cited" or "include the specific files to review" to the description propagates those requirements into every task the main agent delegates to the subagent.

### Defining an Output Format

The single most impactful improvement is a structured output format in the system prompt. It does two things:

- Creates natural stopping points — the subagent knows it's done when every section is filled
- Prevents runaway sessions — without a defined output, subagents struggle to decide when enough is enough

*Example output format for a code reviewer:*

```
Provide your review in the following format:

1. Summary: Brief overview of what was reviewed and overall assessment
2. Critical Issues: Security vulnerabilities, data integrity risks, or logic errors requiring immediate fixes
3. Major Issues: Architecture misalignment, significant quality or performance concerns
4. Minor Issues: Style inconsistencies, documentation gaps, minor optimizations
5. Recommendations: Suggestions, refactoring opportunities, best practices
6. Approval Status: Clear statement — ready to merge, or requires changes
7. Obstacles Encountered: Setup issues, workarounds discovered, commands that needed special flags,
   dependencies or imports that caused problems
```

### Obstacle Reporting

When a subagent discovers a workaround — a dependency issue, a command requiring special flags, an environment quirk — that information must appear in its summary. Without it, the main thread rediscovers the same solutions from scratch, wasting time and tokens.

The fix is simple: add an **"Obstacles Encountered"** section to the output format. Subagents fill in what they're explicitly asked to report.

### Limiting Tool Access

Give each subagent only what it needs:

| Subagent type | Recommended tools |
|---|---|
| Research / read-only | Glob, Grep, Read |
| Code reviewer | Bash (for `git diff`), Glob, Grep, Read |
| Styling / code modifier | Edit, Write (plus read tools) |

Narrow tool access prevents unintended side effects and makes each subagent's role explicit.

---

## When to Use Subagents

The decision rule: **does the intermediate work matter?**

- If no — you just need the result — delegate to a subagent.
- If yes — you need to see and react to what's discovered — keep it in the main thread.

Subagents work best when exploration is separate from execution. If each step depends on what the previous step discovered, that work belongs in your main thread.

### Good Use Cases

**Research and exploration** — The classic case. Investigating how auth works in an unfamiliar codebase? The subagent reads dozens of files and traces function calls in isolation. Your main thread gets: "JWT validation happens in `middleware/auth.js:42`, called from `routes/api.js`."

**Code reviews** — Claude reviews code more effectively when it sees it as authored by someone else. A reviewer subagent runs `git diff` in a fresh context, without the history of how the code was built. You can also encode project-specific review standards in the system prompt for consistent criteria across the team.

**Custom system prompts** — Claude Code's default prompt emphasizes concise, technical responses. Subagents can have completely different instructions:

- *Copywriting subagent* — define tone, audience, and style. The default technical prompt is wrong for landing pages and email campaigns.
- *Styling subagent* — point it at your design system files. Those files load into the subagent's context before it writes a single line of CSS.

### Anti-patterns

**Expert personas** — "You are a Python expert" adds no value. Claude already has that knowledge. There's nothing an "expert" subagent does that the main thread can't.

**Sequential pipelines** — A three-agent flow (reproduce → debug → fix) fails when each step depends on discoveries from the previous one. Information gets lost in handoffs. Bug fixing almost always requires that continuity.

**Test runners** — Test runner subagents hide the output you need to debug failures. "Tests failed" forces you to write additional scripts to get details that would have been visible directly. Testing has shown this pattern performs worst of all common configurations.

---

## Key Benefits

- **Focus** — each subagent concentrates on a single, well-defined task
- **Clean context** — intermediate work stays isolated, preserving main context for longer sessions
- **Concise output** — only the relevant summary comes back, not the noise of the journey
