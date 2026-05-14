import json
import logging
import uuid
from pathlib import Path

import yaml
from pydantic_ai import Agent
from pydantic_core import to_jsonable_python

from llm_agent_template.agent import run_agent

logger = logging.getLogger(__name__)


async def run_pipeline(
    tasks_yaml: str | Path,
    agent: Agent,
) -> None:
    with open(tasks_yaml) as f:
        config: dict = yaml.safe_load(f)

    input_path = Path(config["input_file"])
    output_path = Path(config["output_file"])
    tasks: list[dict] = config["tasks"]

    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_path, "w") as f:
        for task in tasks:
            row = {
                "id": str(uuid.uuid4()),
                "user_message": task["user_message"],
                "image": task.get("image"),
                "metadata": task.get("metadata", {}),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    logger.info("Prepared %d tasks → %s", len(rows), input_path)

    with open(output_path, "w") as f:
        for i, row in enumerate(rows, 1):
            logger.info("[%d/%d] %s...", i, len(rows), row["user_message"][:20])
            result = await run_agent(agent, row["user_message"], image_path=row.get("image"))
            output_row = {
                "id": row["id"],
                "user_message": row["user_message"],
                "image": row["image"],
                "metadata": row["metadata"],
                "output": result.output,
                "messages": to_jsonable_python(result.all_messages),
                "usage": to_jsonable_python(result.usage),
            }
            f.write(json.dumps(output_row) + "\n")

    logger.info("Done. Results saved → %s", output_path)
