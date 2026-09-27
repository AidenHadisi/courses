# AGENTS.md

This repo holds my study notes, one folder per course, viewed in Obsidian.

## The job

I paste a lecture transcript, textbook excerpt, or article. You turn it into a lesson note I can come back to and review quickly.

The note is for **review, not first contact**. Keep what I'd need to recall or apply the idea. Drop everything else. Short but complete — use your judgment on length, not a word count.

## Where it goes

- **New topic** → new file in the course folder, numbered after the last one: `Machine Learning/14. Regularization.md`.
- **Continues an existing topic** → add a new `##` section to that file instead of starting a new one.
- **Unclear course or topic** → ask before writing.

Write the file directly. Then reply with one line: the path and whether you created or appended.

## What to keep, what to cut

Keep:

- Definitions, core ideas, and *why* they work
- Formulas, algorithms, and steps you'd actually use
- Comparisons and trade-offs
- Gotchas and common mistakes the source calls out

Cut:

- Filler and transcript voice ("so basically", "as I said earlier", "in this video")
- Repetition — say each point once, in its best form
- Anecdotes, history, and course logistics unless they carry the idea
- Recap sections that restate the note ("Summary", "Key Takeaways" at the end)

If the source skips a definition I'd need, or states something wrong, add or fix it in one line and mark it *(added)*. Don't pad with extra examples.

## Structure

Every note:

1. Starts with a `#` title, then a 1–3 sentence summary callout:

   ```markdown
   > [!abstract]
   > Gradient descent minimizes a cost function by repeatedly stepping opposite the gradient.
   ```

2. Uses `##` / `###` headings named after the concepts in *this* lesson — not a fixed list. Keep it to two levels when you can.

Beyond that, shape each note to fit its content. A math lecture, a system-design talk, and an opinion article should look different.

## Pick the format that fits the content

| Content | Format |
|---|---|
| Independent points | Bullets, one idea each |
| Steps or a process | Numbered list |
| Two+ things compared on the same attributes | Table |
| A "why" that connects ideas | One or two short sentences |
| Math | LaTeX |
| Code worth reading or running | Fenced code block with a language |
| A flow or relationship easier seen than read | Mermaid diagram |
| A definition, warning, or insight worth interrupting for | Callout — sparingly |

**Bold** a key term where it's introduced, nothing else. `Inline code` for identifiers and commands.

## Density

Bad — reads like the transcript:

```markdown
So the learning rate is really important. If you choose a learning rate that is too small,
gradient descent will work but it will be really slow because it takes tiny baby steps.
On the other hand, if it's too large, you might overshoot the minimum and it may never converge,
or even diverge.
```

Good:

```markdown
**Learning rate** $\alpha$ sets the step size:
- Too small → converges, but slowly
- Too large → overshoots; may never converge or may diverge
```

## Obsidian syntax

- Callouts: type on the first line, every body line prefixed with `> `. One callout per block.

  ```markdown
  > [!warning] Feature scaling
  > Unscaled features make the cost contours elongated, so gradient descent zig-zags.
  ```

  Types I use: `abstract`, `note`, `tip`, `warning`, `example`.
- Math: `$x^2$` inline, `$$...$$` for display. No spaces just inside the `$`. Never `\(...\)`.
- Link to an existing note with `[[Note Name]]` only when it genuinely helps.

## Updating this file

When I correct the style or state a preference, update this file so it sticks.
