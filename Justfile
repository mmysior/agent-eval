run:
    uv run python main.py

compile:
    uv pip compile pyproject.toml --no-annotate --no-header -o requirements.txt
