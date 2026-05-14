# Agent eval

Lightweight eval harness for Pydantic-AI agents.

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
