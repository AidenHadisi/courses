# Claude API

## The Request Lifecycle

Every Claude interaction follows five phases:

1. **Client → your server** — the app sends the user's input
2. **Your server → Anthropic API** — your server forwards the request with your API key
3. **Model processing** — Claude generates a response
4. **Anthropic API → your server** — the response is returned
5. **Your server → client** — your server forwards the text to the UI

---

## Why Requests Must Go Through Your Server

Never call the Anthropic API directly from client-side code. Your API key must be kept secret on the server; exposing it in browser or mobile code lets anyone extract it and make unauthorized requests on your account.

---

## Setup (Python)

Install dependencies:

```bash
pip install anthropic python-dotenv
```

Store your API key in a `.env` file — never hardcode it, and add `.env` to `.gitignore`:

```
ANTHROPIC_API_KEY="your-api-key-here"
```

Initialize the client:

```python
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically
model = "claude-sonnet-4-6"
```

---

## Sending a Request

Use `client.messages.create()` with three required parameters:

| Parameter | Purpose |
|---|---|
| `model` | Which model to use |
| `max_tokens` | Hard ceiling on tokens generated — a safety limit, not a target |
| `messages` | Conversation history as a list of `{role, content}` dicts |

`max_tokens` only caps generation; Claude stops earlier if it naturally finishes. Two roles are valid in `messages`:

- `"user"` — input you send to Claude
- `"assistant"` — responses Claude has generated (used when passing conversation history)

```python
message = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What is quantum computing? Answer in one sentence."}
    ]
)
```

---

## How Claude Processes a Request

### 1. Tokenization

The input text is split into **tokens** — words, subwords, punctuation, or spaces. As a rough approximation, one word ≈ one token.

### 2. Embedding

Each token is converted into an **embedding**: a high-dimensional vector that encodes all possible meanings of that token. Similar words end up with similar vectors.

### 3. Contextualization

The model refines each token's embedding based on surrounding tokens to resolve ambiguity. For example, *quantum* shifts toward "discrete unit of energy" vs. "quantum computing" depending on context. This is the core job of the transformer's attention mechanism.

### 4. Generation

The contextualized embeddings pass through an output layer that produces a probability distribution over the vocabulary. Claude samples from this distribution (not always taking the highest-probability token) to produce natural, varied output. After selecting a token, it appends it to the sequence and repeats from step 3 for the next token.

---

## When Generation Stops

After each token, Claude checks three stopping conditions:

| Condition | Meaning |
|---|---|
| `max_tokens` reached | Hit the limit you specified |
| End-of-sequence token | Claude naturally concluded its response |
| Stop sequence hit | Encountered a predefined stop string |

The `stop_reason` field in the response tells you which condition triggered.

---

## The Response Object

```python
message.content[0].text  # the generated text
```

Full response structure:

```json
{
  "content": [{ "type": "text", "text": "..." }],
  "usage": {
    "input_tokens": 42,
    "output_tokens": 137
  },
  "stop_reason": "end_turn"
}
```

- `content` — list of content blocks; `.text` on the first block gives the response string
- `usage` — token counts for billing and context-window tracking
- `stop_reason` — why generation ended (`end_turn`, `max_tokens`, `stop_sequence`)

---

## Multi-Turn Conversations

Claude is **stateless** — each request is independent with no memory of prior exchanges. To maintain context across turns, you must track conversation history yourself and send the full message list with every request.

> [!IMPORTANT] Every request to Claude starts fresh. If you send only the latest user message, Claude has no knowledge of what was said before.

### Pattern

1. Append the user message to your history list
2. Send the full list to Claude
3. Append Claude's response as an `"assistant"` message
4. Repeat for each turn

```python
def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})

def chat(messages):
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )
    return response.content[0].text

# Usage
messages = []

add_user_message(messages, "Define quantum computing in one sentence.")
answer = chat(messages)
add_assistant_message(messages, answer)

add_user_message(messages, "Write another sentence.")
final_answer = chat(messages)  # Claude sees the full prior context
```

Without appending Claude's previous response as an `"assistant"` message, the follow-up "Write another sentence" would arrive with no context and Claude would respond randomly.

---

## System Prompts

A **system prompt** is a string passed alongside the conversation that instructs Claude how to behave — its role, tone, constraints, and approach. It is separate from the `messages` list and processed before any user input.

```python
client.messages.create(
    model=model,
    messages=messages,
    max_tokens=1000,
    system="You are a patient math tutor. Do not give direct answers — guide students step by step."
)
```

*Example (added):* Without a system prompt, asking "How do I solve 5x + 2 = 3?" gets an immediate complete solution. With the tutor prompt, Claude instead asks guiding questions: *"What operation would isolate x? Think about what you'd do to both sides first."*

> [!NOTE] The `system` parameter is optional. The Anthropic API rejects `system=None`, so omit the key entirely when no system prompt is needed.

### Making `chat()` Accept an Optional System Prompt

```python
def chat(messages, system=None):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
    }
    if system:
        params["system"] = system

    return client.messages.create(**params).content[0].text
```

Usage:

```python
# No persona
answer = chat(messages)

# With persona
answer = chat(messages, system="You are a patient math tutor. Guide, don't solve.")
```

---

## Temperature

**Temperature** (0.0–1.0) controls how Claude samples from the probability distribution during generation. At low values Claude almost always picks the highest-probability token; at high values probability spreads more evenly across candidates, producing more varied output.

| Range | Behavior | Good for |
|---|---|---|
| 0.0–0.3 | Deterministic, consistent | Factual Q&A, coding, data extraction, moderation |
| 0.4–0.7 | Balanced | Summarization, education, constrained creative writing |
| 0.8–1.0 | Creative, varied | Brainstorming, marketing copy, jokes, open-ended writing |

> [!NOTE] Temperature changes the *probability* of varied outputs — it doesn't guarantee them. High temperature Claude may still produce similar responses across runs.

Add it to the `chat()` helper:

```python
def chat(messages, system=None, temperature=1.0):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        params["system"] = system

    return client.messages.create(**params).content[0].text
```

---

## Streaming

By default, Claude generates the full response before sending anything back — which can take 10–30 seconds. Streaming sends text chunks as they are generated, so users see output appear word by word.

### Stream Events

Claude emits a sequence of typed events over a single request:

| Event | Meaning |
|---|---|
| `message_start` | New message beginning |
| `content_block_start` | New content block (text, tool use, etc.) |
| `content_block_delta` | A chunk of generated text ← the payload you display |
| `content_block_stop` | Content block finished |
| `message_delta` | Message-level metadata update |
| `message_stop` | Stream complete |

### Implementation

Use `client.messages.stream()` — the SDK's high-level streaming interface. It handles event parsing and exposes `text_stream`, an iterator of raw text chunks:

```python
with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=messages,
    system=system_prompt,   # optional
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)  # forward each chunk to client

    # After the loop, get the full assembled response for storage/processing
    final_message = stream.get_final_message()
```

`stream.get_final_message()` returns the same response object as a non-streaming call, giving you `usage`, `stop_reason`, etc. once generation is complete.

> [!NOTE] The low-level `stream=True` flag on `messages.create()` also exists, but `messages.stream()` is simpler — it abstracts event parsing and assembles the final message automatically.

---

## Controlling Output Format

By default Claude adds explanatory prose around structured content. For applications that consume raw JSON, code, or lists, this creates parsing friction. Two techniques eliminate it.

### Assistant Message Prefilling + Stop Sequences

Inject a partial assistant message to make Claude think it already started a response in a specific format, then use a stop sequence to cut generation before the closing delimiter:

```python
messages = []
add_user_message(messages, "Generate a short EventBridge rule as JSON.")
add_assistant_message(messages, "```json")  # prefill — Claude continues from here

text = chat(messages, stop_sequences=["```"])  # stop before closing fence
```

What Claude produces (no fences, no prose):

```json
{
  "source": ["aws.ec2"],
  "detail-type": ["EC2 Instance State-change Notification"],
  "detail": { "state": ["running"] }
}
```

Clean it up and parse:

```python
import json
data = json.loads(text.strip())
```

The pattern generalizes: identify what Claude wraps your content in, use that as both the prefill and stop sequence. For code it's a markdown fence; for other formats, adapt accordingly.

> [!WARNING] `stop_sequences` must be added to your `chat()` helper's `params` dict (like `system`) for this to work.

---

## Prompt Engineering vs. Prompt Evaluation

**Prompt engineering** is the practice of crafting prompts that reliably produce the output you want — using techniques like few-shot examples, XML structure, role assignment, and output constraints.

**Prompt evaluation** is how you *measure* whether a prompt works — automated testing against expected outputs, comparing prompt versions, and scoring across a representative set of inputs.

### The Testing Trap

After writing a prompt, engineers typically take one of three paths:

| Approach | Risk |
|---|---|
| Test once, ship | Breaks on unexpected real-world inputs |
| Test a few cases, patch edge cases | Better, but users will always find more |
| Run an evaluation pipeline, iterate on metrics | Catches weaknesses before production |

The first two feel fast but accumulate hidden risk. Real users interact with prompts in ways you never anticipate during development.

> [!IMPORTANT] Build evaluation before you ship. Defining what "correct" looks like and measuring it objectively is what separates prototypes from reliable production systems.

### Evaluation Workflow

A standard eval pipeline has five steps:

1. **Draft a prompt** — write the template you want to measure, with placeholders for dynamic input:
   ```python
   prompt = f"Please answer the user's question:\n\n{question}"
   ```

2. **Build an eval dataset** — a list of representative inputs that cover expected and edge-case usage. Start small (tens of examples); scale to hundreds or thousands as needed. You can write these by hand or generate them with Claude.

3. **Run the prompt** — interpolate each input into the template and send it to Claude. Collect all responses.

4. **Grade the responses** — feed each (question, response) pair to a grader (another Claude call works well) that scores output quality, typically 1–10. Average across the dataset to get a single score.
   ```
   Example scores: math (10) + oatmeal (4) + Moon distance (9) → avg 7.67
   ```

5. **Iterate** — modify the prompt (add detail, restructure, tighten constraints), re-run, compare scores. Keep the version with the highest average.

The loop turns prompt improvement from intuition into measurement. A score of 7.67 → 8.7 after adding `"Answer with ample detail"` is an objective signal the change helped.

### Building an Eval Pipeline (Worked Example)

**Goal:** evaluate a prompt that returns clean Python, JSON, or regex for AWS tasks — no prose, no headers.

**Starting prompt (v1):**

```python
prompt = f"Please provide a solution to the following task:\n{task}"
```

#### Generating the Dataset

Use Claude to generate test cases rather than writing them by hand. Use a faster/cheaper model (Haiku) for data generation since output quality doesn't matter here:

```python
def generate_dataset():
    generation_prompt = """
Generate an evaluation dataset for prompts that produce Python, JSON, or Regex for AWS tasks.
Return a JSON array of objects with a single "task" key each.

Example output:
```json
[{"task": "Description of task"}, ...]
```

* Each task should be solvable with one function, one JSON object, or one regex.
* Keep tasks small in scope.

Generate 3 objects.
"""
    messages = []
    add_user_message(messages, generation_prompt)
    add_assistant_message(messages, "```json")           # prefill
    text = chat(messages, stop_sequences=["```"])        # stop before closing fence
    return json.loads(text)

dataset = generate_dataset()

# Persist so you don't regenerate on every run
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

The prefill + stop sequence pattern (covered above) ensures clean JSON output that `json.loads()` can parse directly.

#### Running the Pipeline

Three functions compose the eval loop:

```python
def run_prompt(test_case):
    """Merge the prompt template with a test case and return Claude's output."""
    prompt = f"Please solve the following task:\n\n{test_case['task']}"
    messages = []
    add_user_message(messages, prompt)
    return chat(messages)

def run_test_case(test_case):
    """Run one test case and return output + score."""
    output = run_prompt(test_case)
    score = 10  # placeholder — replaced by a grader in the next step
    return {"test_case": test_case, "output": output, "score": score}

def run_eval(dataset):
    """Run all test cases and collect results."""
    return [run_test_case(tc) for tc in dataset]
```

Execute against the saved dataset:

```python
with open("dataset.json") as f:
    dataset = json.load(f)

results = run_eval(dataset)
print(json.dumps(results, indent=2))
```

Each result has three fields: `test_case` (the input), `output` (Claude's response), and `score`. At this stage the prompt has no formatting instructions, so `output` will contain verbose prose — that's expected. Grading logic and prompt refinement come next.

#### Graders

A grader takes a (test_case, output) pair and returns a score (1–10). Three approaches:

| Type | How it works | Best for |
|---|---|---|
| **Code** | Programmatic checks (length, regex validation, JSON parsing) | Objective, verifiable criteria |
| **Model** | Another Claude call that scores the response | Subjective quality (helpfulness, task-following) |
| **Human** | Manual review | Final validation, ambiguous criteria |

For the AWS code generation example, the criteria map naturally:

| Criterion | Grader type |
|---|---|
| Format — only Python/JSON/regex, no prose | Code grader |
| Valid syntax | Code grader |
| Correctly solves the task | Model grader |

**Model grader implementation.** Ask for `strengths`, `weaknesses`, and `reasoning` alongside the score — without reasoning, models default to middling scores:

```python
def grade_by_model(test_case, output):
    eval_prompt = f"""
You are an expert code reviewer. Evaluate this AI-generated solution.

Task: {test_case['task']}
Solution: {output}

Return a JSON object with:
- "strengths": array of 1-3 key strengths
- "weaknesses": array of 1-3 areas for improvement
- "reasoning": concise explanation
- "score": integer 1-10
"""
    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")
    return json.loads(chat(messages, stop_sequences=["```"]))
```

**Wire grading into the pipeline:**

```python
from statistics import mean

def run_test_case(test_case):
    output = run_prompt(test_case)
    grade = grade_by_model(test_case, output)
    return {
        "test_case": test_case,
        "output": output,
        "score": grade["score"],
        "reasoning": grade["reasoning"],
    }

def run_eval(dataset):
    results = [run_test_case(tc) for tc in dataset]
    print(f"Average score: {mean(r['score'] for r in results)}")
    return results
```

> [!NOTE] Model graders can be inconsistent across runs. Use them as a directional signal, not an absolute truth — a score improvement of 0.5–1.0 points is meaningful; noise within that range is not.

**Code grader implementation.** Syntax validation is binary — either it parses or it doesn't:

```python
import ast, re

def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0

def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0

def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0

def grade_syntax(output, test_case):
    fmt = test_case.get("format")
    if fmt == "json":    return validate_json(output)
    if fmt == "python":  return validate_python(output)
    if fmt == "regex":   return validate_regex(output)
    return 0
```

Add a `"format"` field to each dataset entry so the grader knows which validator to apply:

```json
{"task": "Create a Python function to validate an AWS IAM username", "format": "python"}
```

Update `generate_dataset()` to include this field in the generation prompt's example output.

**Combining scores** — average the model and syntax scores for a single composite metric:

```python
def run_test_case(test_case):
    output = run_prompt(test_case)
    model_grade = grade_by_model(test_case, output)
    syntax_score = grade_syntax(output, test_case)
    score = (model_grade["score"] + syntax_score) / 2
    return {
        "test_case": test_case,
        "output": output,
        "score": score,
        "reasoning": model_grade["reasoning"],
    }
```

Adjust the weighting if one criterion matters more for your use case. The absolute score value is less important than whether it moves when you change the prompt.
