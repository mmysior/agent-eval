import asyncio
import logging
import sys

from agent_eval import get_agent, run_eval
from agent_eval.core.config import config
from agent_eval.core.logging import setup_logging
from agent_eval.tools import tools

logger = logging.getLogger(__name__)

agent_with_tools = get_agent(tools=tools)
agent_no_tools = get_agent()


async def main() -> None:
    input_files = sorted(config.input_path.glob("*.jsonl"))
    if not input_files:
        logger.error("No input files found in %s. Run `agent-eval prepare-input` first.", config.input_path)
        sys.exit(1)

    for input_file in input_files:
        stem = input_file.stem
        await run_eval(agent_with_tools, input_file, config.output_path / f"{stem}_with_tools.jsonl")
        await run_eval(agent_no_tools, input_file, config.output_path / f"{stem}_no_tools.jsonl")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
