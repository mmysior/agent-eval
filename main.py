import asyncio
import sys

from llm_agent_template.agent import get_agent
from llm_agent_template.core.config import config
from llm_agent_template.core.logging import setup_logging
from llm_agent_template.pipeline import run_pipeline
from llm_agent_template.tools.calc import (
    calculate_power,
    calculate_transmission,
    convert_speed,
)


async def main() -> None:
    setup_logging()

    if len(sys.argv) != 2:
        print("Usage: uv run main.py <path/to/tasks.yaml>")
        sys.exit(1)

    agent = get_agent(
        provider=config.DEFAULT_PROVIDER,
        model=config.DEFAULT_MODEL,
        prompt_name="agent",
        tools=(convert_speed, calculate_power, calculate_transmission),
    )

    await run_pipeline(sys.argv[1], agent)


if __name__ == "__main__":
    asyncio.run(main())
