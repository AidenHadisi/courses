# MCP (Model Context Protocol)

MCP is an open protocol that standardizes how AI applications connect to external tools and data sources. A **server** exposes capabilities (tools, resources, prompts); a **client** connects to servers and surfaces those capabilities to a model. Your application code is the client — you build a server for your domain and wire it into any MCP-compatible host (Claude Desktop, VS Code, etc.).

## The Three Server Primitives

Each primitive is owned by a different layer of the stack:

| Primitive | Controlled by | Use when you need to… |
|---|---|---|
| **Tools** | The model (Claude) | Give Claude new capabilities to use autonomously |
| **Resources** | Application code | Fetch data for UI elements or inject context into prompts |
| **Prompts** | The user | Provide predefined, optimized workflows users trigger on demand |

### Tools — Model-Controlled

Claude decides when to call tools and uses the results directly to accomplish a task. Examples: running code, querying a database, calling an external API. The app doesn't need to orchestrate anything — Claude handles the decision.

### Resources — App-Controlled

Your application code fetches resource data and decides how to use it — typically to populate autocomplete UI or augment the prompt with additional context before sending it to Claude. The model never requests resources directly.

Common patterns:
- Populate a document picker with a `list_docs` resource
- Inject file contents into the prompt when a user `@mention`s a document

### Prompts — User-Controlled

Prompts are parameterized message templates users trigger explicitly (slash commands, buttons, menu items). The server renders the full message list with variables filled in; the client sends it to Claude.

**Decision guide:**
- Giving Claude new abilities → **Tool**
- Getting data into the app or adding context → **Resource**
- Creating a user-triggered workflow → **Prompt**

---

## Building an MCP Server with the Python SDK

The Python MCP SDK simplifies server creation by using decorators and type hints instead of manual JSON schema writing.

### Setup

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocumentMCP", log_level="ERROR")
```

### Defining Tools

Use the `@mcp.tool` decorator with Python type hints and Pydantic `Field` for parameter descriptions. The SDK auto-generates schemas from these.

```python
from pydantic import Field

@mcp.tool(
    name="read_doc_contents",
    description="Read the contents of a document and return it as a string."
)
def read_document(
    doc_id: str = Field(description="Id of the document to read")
):
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    return docs[doc_id]
```

```python
@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing a string in the document's content."
)
def edit_document(
    doc_id: str = Field(description="Id of the document to edit"),
    old_str: str = Field(description="Text to replace. Must match exactly, including whitespace."),
    new_str: str = Field(description="Replacement text.")
):
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    docs[doc_id] = docs[doc_id].replace(old_str, new_str)
    return docs[doc_id]   # return updated content as confirmation
```

### Key Benefits

- No manual JSON schema writing
- Type hints provide automatic validation
- `Field` descriptions help Claude understand parameters
- Tool registration is automatic via decorators
- Python exceptions integrate naturally as error handling

---

## Building an MCP Client

### Architecture

In real-world projects you typically build either a client or a server, not both. The client side has two components:

- **Client Session** — the actual connection to the server (from the MCP Python SDK); requires careful resource management
- **MCP Client** — a custom wrapper class that manages the session lifecycle and exposes a clean interface

### Application Flow

```
CLI code → MCP Client → Client Session → MCP Server
```

The client is used at two points:
1. Fetch the list of available tools to send to Claude
2. Execute a tool when Claude requests it

### Core Methods

```python
async def list_tools(self) -> list[types.Tool]:
    result = await self.session().list_tools()
    return result.tools
```

```python
async def call_tool(
    self, tool_name: str, tool_input: dict
) -> types.CallToolResult | None:
    return await self.session().call_tool(tool_name, tool_input)
```

`tool_name` and `tool_input` come directly from Claude's tool call response.

```python
async def list_prompts(self) -> list[types.Prompt]:
    result = await self.session().list_prompts()
    return result.prompts
```

```python
async def get_prompt(self, prompt_name: str, args: dict[str, str]):
    result = await self.session().get_prompt(prompt_name, args)
    return result.messages
```

`get_prompt` handles variable interpolation — `args` becomes keyword arguments passed into the prompt function on the server. For example, `{"doc_id": "plan.md"}` fills `{doc_id}` in the template and returns the fully rendered message list.

### Testing

```bash
uv run mcp_client.py   # verify client connects and lists tools
uv run main.py         # run the full application
```

### End-to-End Flow

When a user asks a question (e.g. *"What is in report.pdf?"*):

1. Client calls `list_tools()` → tools sent to Claude with the user's message
2. Claude responds with a tool call (`read_doc_contents`, `doc_id="report.pdf"`)
3. Client calls `call_tool()` → result returned to Claude
4. Claude uses the result to answer the user

---

## MCP Resources

Resources expose read-only data from the server — analogous to GET endpoints. Use them to fetch information rather than perform actions.

### Two Resource Types

**Direct** — static URI, no parameters:

```python
@mcp.resource(
    "docs://documents",
    mime_type="application/json"
)
def list_docs() -> list[str]:
    return list(docs.keys())
```

**Templated** — URI with `{param}` placeholders; the SDK parses them and passes as keyword args:

```python
@mcp.resource(
    "docs://documents/{doc_id}",
    mime_type="text/plain"
)
def fetch_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    return docs[doc_id]
```

### How It Works

```
app code → MCP Client → ReadResourceRequest(uri) → MCP Server → ReadResourceResult
```

The SDK auto-serializes return values — return a `list`, `dict`, or `str` and the SDK handles conversion.

### `mime_type` Hints

| Value | Use for |
|---|---|
| `application/json` | structured data |
| `text/plain` | plain text |
| `application/pdf` | binary files |

### Reading Resources (Client Side)

```python
import json
from typing import Any
from pydantic import AnyUrl

async def read_resource(self, uri: str) -> Any:
    result = await self.session().read_resource(AnyUrl(uri))
    resource = result.contents[0]

    if isinstance(resource, types.TextResourceContents):
        if resource.mimeType == "application/json":
            return json.loads(resource.text)

    return resource.text
```

- `result.contents` is a list; index `[0]` is sufficient when fetching a single resource
- JSON MIME type → parse and return a Python object; otherwise return raw text

### Use Case: Document Mentions

Resources are ideal for `@mention` features. When a user types `@report.pdf`, the app:

1. Calls `list_docs` resource to populate autocomplete
2. Calls `fetch_doc` resource to get the contents via `read_resource(uri)`
3. Injects the contents directly into the prompt — no tool call needed; Claude has immediate context

---

## MCP Prompts

Prompts are pre-built, expert-crafted instruction templates that clients expose as slash commands (e.g. `/format`). Users get better results from a well-tested prompt than from writing their own from scratch.

### Why Use Prompts

Users can already ask Claude to do most tasks directly — but a prompt you've carefully crafted, tested, and tuned for edge cases will consistently outperform ad-hoc instructions. As the server author, you encode domain expertise once; users benefit automatically.

### Defining Prompts

Use the `@mcp.prompt` decorator with a `Field`-annotated signature. The function returns a list of `base.Message` objects sent directly to Claude.

```python
from mcp.server.fastmcp import FastMCP
from mcp import base
from pydantic import Field

@mcp.prompt(
    name="format",
    description="Rewrites the contents of the document in Markdown format."
)
def format_document(
    doc_id: str = Field(description="Id of the document to format")
) -> list[base.Message]:
    prompt = f"""
Your goal is to reformat a document to be written with markdown syntax.

The id of the document you need to reformat is:
<document_id>
{doc_id}
</document_id>

Add in headers, bullet points, tables, etc as necessary. Feel free to add structure.
Use the 'edit_document' tool to edit the document.
"""
    return [base.UserMessage(prompt)]
```

- Return a **list of messages** — multiple user/assistant turns are supported for more complex flows
- Variables are interpolated into the template at call time
- Prompt names become the slash command the user types in the client

### Key Benefits

| Benefit | Details |
|---|---|
| **Consistency** | Same high-quality result every time |
| **Expertise** | Encode domain knowledge and edge-case handling once |
| **Reusability** | Any client application can use the same prompts |
| **Maintainability** | Update in one place; all clients benefit immediately |

### Testing Prompts

Use the MCP Inspector (same as tools) to verify prompts before deploying. The inspector shows the fully interpolated message list so you can confirm variable substitution and prompt structure before users rely on it.

---

## Transports

A transport is the communication channel used to exchange JSON messages between client and server.

### Stdio Transport

The client launches the server as a subprocess and communicates over standard streams:

- **Client → Server:** write JSON to the server's `stdin`
- **Server → Client:** write JSON to `stdout`
- Either side can initiate a message at any time
- Both processes must run on the **same machine**

Stdio is the simplest transport and the default for local development. You can test a server directly from the terminal — run it with `uv run server.py`, then paste raw JSON messages into stdin and observe the responses.

### Connection Handshake

Every MCP connection begins with a mandatory three-message sequence before any other messages can be exchanged:

```
Client → Initialize Request
Server → Initialize Result
Client → Initialized Notification  (no response)
```

Only after this handshake are tool calls, resource reads, and prompt requests valid.

### Four Communication Patterns

Stdio handles all four patterns cleanly because both streams are always open:

| Direction | Channel |
|---|---|
| Client → Server request | Client writes to server stdin |
| Server → Client response | Server writes to stdout |
| Server → Client request | Server writes to stdout |
| Client → Server response | Client writes to server stdin |

### Stdio vs Other Transports

Stdio represents the "ideal" case — fully bidirectional with no constraints. HTTP-based transports introduce limitations (e.g. the server cannot always initiate requests to the client), so stdio is the baseline for understanding what complete MCP communication looks like.

**Use stdio for:** local development, testing, desktop integrations  
**Use HTTP-based transports for:** remote servers, multi-client deployments, cloud hosting

### Streamable HTTP Transport

Enables clients to connect to remotely hosted MCP servers over HTTP. The fundamental challenge: HTTP is client-initiated — servers have known URLs, clients do not. This means servers cannot easily push messages to clients.

**Two settings that restrict behavior:**

| Setting | Default | Effect when `True` |
|---|---|---|
| `stateless_http` | `False` | No session IDs, no GET SSE channel — fully stateless |
| `json_response` | `False` | POST responses return plain JSON instead of SSE |

#### `stateless_http` — Scaling Motivation

The default stateful mode requires each client to maintain **two connections** to the same server instance: a long-lived GET SSE connection plus individual POST requests for tool calls. This works fine on a single server, but breaks under horizontal scaling.

With a load balancer routing requests across multiple server instances, a client's GET SSE connection might land on Server A while a tool-call POST lands on Server B. Server B would need to coordinate with Server A to deliver server-initiated messages — a complex distributed systems problem.

`stateless_http=True` eliminates this by removing session state entirely: no session IDs are issued, no GET SSE channel is created, and each request is handled independently by whichever instance receives it. As a side effect, the initialization handshake is no longer required — clients can make requests directly.

**What you lose:**

- Progress notifications
- Logging notifications
- Server-initiated sampling (`Create Message` requests)
- `List Roots` requests
- `Initialized` / `Cancelled` notifications

#### `json_response` — Simpler POST Responses

`json_response=True` is narrower in scope: it only affects **POST request responses**. Instead of streaming intermediate SSE messages as a tool executes, the client receives a single plain JSON response with the final result. The GET SSE channel is unaffected.

**What you lose:**
- Intermediate progress messages during tool execution
- Log messages emitted during tool execution

**Decision guide:**

| Scenario | Recommendation |
|---|---|
| Local or single-instance deployment | Keep both `False` (full functionality) |
| Horizontally scaled / load-balanced | Set `stateless_http=True`; design tools without server-initiated messages |
| System expecting plain JSON | Set `json_response=True` |
| Serverless platform | Set both `True`; server becomes pure request/response |

> [!WARNING] If you develop locally with stdio but plan to deploy over HTTP, test with HTTP during development. The behavior difference between stateful and stateless modes is significant and will surface bugs that stdio testing won't catch.

#### How StreamableHTTP Enables Server-to-Client Communication

StreamableHTTP uses **Server-Sent Events (SSE)** to work around HTTP's client-initiated constraint.

**1. Initialization**

```
Client → POST /mcp   Initialize Request
Server → 200         Initialize Result  +  mcp-session-id: <id>
Client → POST /mcp   Initialized Notification  (includes session ID header)
```

The `mcp-session-id` must be included in every subsequent request so the server can route responses back to the correct client.

**2. Primary SSE Connection**

After init, the client opens a persistent GET connection:

```
Client → GET /mcp  (long-lived)
Server ← streams messages at any time via this channel
```

This is the channel the server uses for **server-initiated requests** (sampling, list roots) and some notifications.

**3. Dual SSE on Tool Calls**

Each tool call creates a second, short-lived SSE connection:

| Connection | Lifetime | Carries |
|---|---|---|
| Primary SSE | Indefinite | Server-initiated requests, progress notifications |
| Tool SSE | Until result sent | Logging messages, tool result |

**4. What breaks without SSE**

- `stateless_http=True` — eliminates the GET SSE channel entirely; all server-to-client push (progress, logging, sampling) stops working
- `json_response=True` — POST responses return plain JSON; intermediate SSE messages (progress, logging) during tool calls are lost, but the GET SSE channel remains

---

## MCP Message Protocol

All MCP communication is JSON. The authoritative definition of every message type lives in the MCP specification repository on GitHub (written in TypeScript for clarity, not execution).

### Two Categories

**Request-Result pairs** — always a matched request + response:

| Request | Result |
|---|---|
| Call Tool Request | Call Tool Result |
| List Prompts Request | List Prompts Result |
| Read Resource Request | Read Resource Result |
| Initialize Request | Initialize Result |

**Notifications** — one-way, no response expected:

- Progress Notification
- Logging Message Notification
- Tool List Changed Notification
- Resource Updated Notification

### Bidirectional Protocol

Both clients and servers can initiate messages — servers are not purely reactive. Server-to-client messages include sampling requests and notifications (progress, logging, resource updates).

This matters when choosing a transport: some transports restrict which direction messages can flow. The streamable HTTP transport, for example, has limitations on server-initiated messages that stdio does not.

---

## Roots

Roots grant an MCP server access to specific directories on the local file system. They solve two problems at once: they give Claude the context needed to locate files without requiring users to type full paths, and they limit access so the server can't touch files outside approved locations.

### The Problem

When a user says "convert biking.mp4", Claude only has the filename. Without roots, it has no way to locate the file on a complex file system. Requiring users to provide full paths is a poor experience.

### Workflow With Roots

1. User asks to convert a file
2. Claude calls `list_roots` → gets the approved directory list
3. Claude calls `read_dir` on those directories → finds the file
4. Claude calls the tool with the resolved full path

This is fully automatic — the user never types a path.

### Access Control

Roots act as a permission boundary. If only `~/Movies` is granted, attempts to access `~/Documents` return an error. The MCP SDK does **not** enforce this automatically — you must implement it yourself.

Typical pattern:

```python
async def is_path_allowed(path: str, ctx: Context) -> bool:
    roots = await ctx.session.list_roots()
    resolved = Path(path).resolve()
    return any(
        resolved.is_relative_to(Path(root.uri.removeprefix("file://")).resolve())
        for root in roots.roots
    )
```

Call this at the start of any tool that touches the file system before performing the operation:

```python
@mcp.tool()
async def convert_video(file_path: str, *, ctx: Context):
    if not await is_path_allowed(file_path, ctx):
        raise PermissionError(f"Access to {file_path} is not permitted")
    # proceed with conversion
```

### Key Benefits

- **User-friendly** — no full paths required from the user
- **Focused search** — Claude only scans approved directories
- **Security** — sensitive files outside approved roots are unreachable
- **Flexible injection** — roots can be provided via tools or injected directly into prompts

---

## Logging and Progress Notifications

Tools that take time to run (research, data processing, etc.) can emit real-time status messages and progress updates to the client. Without these, users see nothing until the operation finishes.

### Server Side

Add `context: Context` as a **keyword-only** argument (after `*`). The SDK injects it automatically — do not include it in `Field` definitions.

```python
from mcp import Context
from pydantic import Field

@mcp.tool(
    name="research",
    description="Research a given topic"
)
async def research(
    topic: str = Field(description="Topic to research"),
    *,
    context: Context
):
    await context.info("Starting research...")
    await context.report_progress(20, 100)
    sources = await do_research(topic)

    await context.info("Writing report...")
    await context.report_progress(70, 100)
    return await generate_report(sources)
```

Key methods on `context`:

| Method | Purpose |
|---|---|
| `context.info(msg)` | Send a log message to the client |
| `context.report_progress(current, total)` | Emit a progress update |

### Client Side

Supply callbacks when creating the session and when calling tools:

```python
from mcp.types import LoggingMessageNotificationParams

async def logging_callback(params: LoggingMessageNotificationParams):
    print(params.data)

async def progress_callback(progress: float, total: float | None, message: str | None):
    if total is not None:
        print(f"Progress: {progress}/{total} ({progress / total * 100:.1f}%)")
    else:
        print(f"Progress: {progress}")

async with ClientSession(read, write, logging_callback=logging_callback) as session:
    await session.initialize()
    await session.call_tool(
        name="research",
        arguments={"topic": "quantum computing"},
        progress_callback=progress_callback,
    )
```

- `logging_callback` → passed to `ClientSession`, applies to the whole session
- `progress_callback` → passed per `call_tool` call

### Presentation

Both callbacks are optional. Present however suits your app:

- **CLI** — print directly to terminal
- **Web** — push via WebSockets or server-sent events
- **Desktop** — update progress bars and status labels

---

## Sampling

Sampling lets an MCP server request text generation through the connected client, rather than calling Claude directly. The server delegates the LLM call to the client, which already has credentials and an active connection.

### Why Use It

Without sampling, a server that needs to generate text must manage its own API key, authentication, and token costs. With sampling, the server just asks "please call Claude for me" — the client handles everything.

**Ideal for public servers:** you don't want arbitrary users running up AI costs on your API key. Sampling shifts each user's token costs to their own client.

### Flow

```
Server (tool runs) → create_message() → Client → Claude API → result back to server
```

### Server Side

Add `ctx: Context` to your tool signature, then call `ctx.session.create_message()`:

> [!NOTE] `ctx` here is a positional argument, unlike the keyword-only `context` pattern used in logging/progress tools. Both work; be consistent within your codebase.

```python
from mcp import Context
from mcp.types import SamplingMessage, TextContent

@mcp.tool()
async def summarize(text_to_summarize: str, ctx: Context):
    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=f"Summarize:\n{text_to_summarize}")
            )
        ],
        max_tokens=4000,
        system_prompt="You are a helpful research assistant",
    )

    if result.content.type == "text":
        return result.content.text
    raise ValueError("Sampling failed")
```

### Client Side

Define a callback and pass it when creating the `ClientSession`:

```python
from mcp.types import CreateMessageRequestParams, CreateMessageResult, TextContent
from mcp.shared.context import RequestContext

async def sampling_callback(
    context: RequestContext, params: CreateMessageRequestParams
) -> CreateMessageResult:
    text = await chat(params.messages)   # your existing Claude call
    return CreateMessageResult(
        role="assistant",
        model=model,
        content=TextContent(type="text", text=text),
    )

async with ClientSession(read, write, sampling_callback=sampling_callback) as session:
    await session.initialize()
```

### Summary

| Without sampling | With sampling |
|---|---|
| Server needs its own API key | No credentials on the server |
| Server pays token costs | Client pays token costs |
| More server complexity | Client handles LLM integration |

---

## Testing with the MCP Inspector

The Python MCP SDK includes a built-in browser-based inspector for testing and debugging without connecting to a full application.

### Starting the Inspector

```bash
mcp dev mcp_server.py
```

Opens at `http://127.0.0.1:6274` by default.

### Workflow

1. Open the URL in a browser
2. Click **Connect** to initialize the server
3. Navigate to the **Tools** tab → click **List Tools**
4. Select a tool, fill in parameters, click **Run Tool**
5. Verify the result and returned data

### Key Features

- Server state persists between tool calls — edits made with one tool are visible when you call another
- Test edge cases and error conditions directly
- No separate test scripts needed
- Immediate feedback loop for iterating on tool implementations
