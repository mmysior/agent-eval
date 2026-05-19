run:
    uv run python main.py

mcp:
    uv run python -m agent_eval.servers.mcp

prepare tasks_yaml:
    uv run agent-eval prepare-input {{tasks_yaml}}
