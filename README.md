# Agent eval

Lightweight eval harness for Pydantic-AI agents.

## Quickstart

**1. Fork and clone this repo, then install dependencies:**

```bash
git clone https://github.com/your-org/agent-eval
cd agent-eval
uv sync
cp .env.example .env
```

**2. Configure your provider in `.env`:**

If you have access to a cloud API:

```env
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

Or use **LM Studio** to run models locally for free — no API key needed:

1. Download [LM Studio](https://lmstudio.ai) and install `google/gemma-4-e4b` (or any model with tool calling support)
2. Start the local server in LM Studio (**Local Server** tab → **Start Server**)
3. Set in `.env`:

```env
DEFAULT_PROVIDER=lmstudio
DEFAULT_MODEL=google/gemma-4-e4b
```

**3. Write your tools** in `src/agent_eval/tools/` — each tool is a plain async function with type annotations and a docstring:

```python
async def my_tool(param: str) -> str:
    """Description of what this tool does."""
    return f"result: {param}"
```

For non-trivial logic, keep the tool thin and put the actual computation in `src/agent_eval/functions/` as a pure (non-async) function — then call it from the tool. This separation makes the logic easy to test independently:

```python
# src/agent_eval/functions/my_module.py
def compute(param: str) -> str:
    return f"result: {param}"

# src/agent_eval/tools/my_module.py
from agent_eval.functions.my_module import compute

async def my_tool(param: str) -> str:
    """Description of what this tool does."""
    return compute(param)
```

Export all tools from `src/agent_eval/tools/__init__.py`:

```python
from agent_eval.tools.my_module import my_tool
tools = (my_tool,)
```

**4. Define your experiment** in `tasks/my_experiment.yaml` — see `tasks/_template.yaml` for the format.

**5. Generate input and run:**

```bash
uv run agent-eval prepare-input tasks/my_experiment.yaml
uv run python main.py
```

Results land in `data/output/` as paired JSONL files — one run with your tools, one without.

---

## Workflow

1. Define experiment scenarios in `tasks/*.yaml`
2. Generate input JSONL: `uv run agent-eval prepare-input tasks/my_experiment.yaml`
3. Inspect `data/input/my_experiment.jsonl` before spending API credits
4. Run `python main.py` — iterates all input files, runs each with and without tools

Results land in `data/output/` as paired files:
```
data/output/my_experiment_with_tools.jsonl
data/output/my_experiment_no_tools.jsonl
```

Runs are resume-safe — already completed rows are skipped automatically.

## Structure

```
tasks/        # experiment definitions (YAML)
data/input/   # generated input JSONL
data/output/  # results JSONL
prompts/      # system prompts
src/
  agent_eval/
    tools/    # your tools go here
main.py       # wire up your agents here
```

## YAML format

See `tasks/_template.yaml`. The `metadata` field is yours — use it for categories, tags, or expected outputs for downstream analysis.

## Tools

Add your tools to `src/agent_eval/tools/`. Each tool is a plain async Python function with type annotations and a docstring — Pydantic-AI uses these to generate the tool schema automatically.

Export all tools from `src/agent_eval/tools/__init__.py`:

```python
from agent_eval.tools.my_module import my_tool_a, my_tool_b

tools = (my_tool_a, my_tool_b)
```

Then in `main.py`, pass them to the agent explicitly:

```python
from agent_eval.tools import tools

agent_with_tools = get_agent(tools=tools)
agent_no_tools   = get_agent()
```

This gives you a direct comparison between the agent with and without tools on the same scenarios.

## Output schema

Both input and output files are JSONL — one JSON object per line. The schemas are defined as Pydantic models in `src/agent_eval/schemas.py` and can be used directly for parsing results:

```python
from agent_eval import InputRow, OutputRow

with open("data/output/my_experiment_with_tools.jsonl") as f:
    rows = [OutputRow.model_validate_json(line) for line in f]
```

### InputRow

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique row ID (SHA256 of message + image + repetition index) |
| `scenario_id` | `str` | Shared across repetitions of the same task |
| `model` | `str` | Model name from the task YAML |
| `provider` | `str` | Provider name from the task YAML |
| `model_settings` | `dict` | Temperature, max tokens, etc. |
| `user_message` | `str` | The prompt sent to the agent |
| `image` | `str \| None` | Optional path to an image attachment |
| `metadata` | `dict` | Arbitrary metadata from the task YAML |

### OutputRow

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Matches the input row ID |
| `output` | `str \| None` | Final agent response, or `None` on error |
| `all_messages` | `list[ModelMessage]` | Full message history including tool calls |
| `usage` | `RunUsage \| None` | Token usage from `pydantic_ai.usage.RunUsage`, or `None` on error |
| `error` | `str \| None` | Error message if the run failed, otherwise `None` |

`usage` is a `pydantic_ai.usage.RunUsage` instance — see the [Pydantic AI docs](https://ai.pydantic.dev) for its fields (`requests`, `input_tokens`, `output_tokens`, `tool_calls`, etc.).

## Running evaluations in Docker

For long-running eval batches, Docker lets you fire off the whole run in the background and walk away.

First, generate the input files locally:

```bash
uv run agent-eval prepare-input tasks/*.yaml
```

Then launch the container:

```bash
docker compose up -d
```

The container runs `main.py` against all files in `data/input/` and writes results to `data/output/`. The container exits when the batch is complete.

Follow logs at any time:

```bash
docker compose logs -f
```

## MCP Server

Tools are also exposed as an MCP server over Streamable HTTP for testing with external clients (e.g. LM Studio):

```bash
just mcp
# or
uv run python -m agent_eval.servers.mcp
# server listens on http://localhost:8000/mcp
```

### Adding to LM Studio

1. Start the MCP server (see above)
2. In LM Studio, open **Settings → MCP Servers** and edit `mcp.json`
3. Add the following entry:

```json
{
  "mcpServers": {
    "agent-eval": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

4. Reload LM Studio — the tools will appear in the model's tool list.
