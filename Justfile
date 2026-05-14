run:
    uv run python main.py

mcp:
    uv run python -m agent_eval.servers.mcp

compile:
    uv pip compile pyproject.toml --no-annotate --no-header -o requirements.txt
