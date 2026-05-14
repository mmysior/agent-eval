import asyncio
import sys

from llm_agent_template.core.config import config
from llm_agent_template.core.logging import setup_logging
from llm_agent_template.pipeline import agent_from_yaml, prepare, run


def main() -> None:
    setup_logging()

    yaml_files = sorted(f for f in config.tasks_path.glob("*.yaml") if not f.name.startswith("_"))
    if not yaml_files:
        print(f"No task files found in {config.tasks_path}", file=sys.stderr)
        sys.exit(1)

    for f in yaml_files:
        input_file = config.input_path / (f.stem + ".jsonl")
        if not input_file.exists():
            name, count = prepare(f)
            print(f"Prepared {count} tasks -> {config.input_path / name}")

    for f in yaml_files:
        try:
            agent, tool_defs = agent_from_yaml(f)
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"Error loading tools from {f.name}: {e}", file=sys.stderr)
            sys.exit(1)

        succeeded, failed = asyncio.run(run(f.stem + ".jsonl", agent, tool_defs))
        print(f"{f.stem}: {succeeded} ok, {failed} failed")


if __name__ == "__main__":
    main()
