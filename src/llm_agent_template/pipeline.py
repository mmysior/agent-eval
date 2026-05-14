import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic_ai import Agent
from pydantic_core import to_jsonable_python

from llm_agent_template.agent import run_agent
from llm_agent_template.core.config import config


def prepare(tasks_yaml: str | Path) -> tuple[Path, int]:
    with open(tasks_yaml) as f:
        cfg: dict = yaml.safe_load(f)

    input_path = config.input_path / cfg["file"]
    input_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = cfg["tasks"]
    n: int = cfg.get("n", 1)
    provider: str = cfg.get("provider", "")
    model: str = cfg.get("model", "")
    prompt_name: str = cfg.get("prompt_name", "agent")
    model_settings: dict = cfg.get("model_settings") or {}
    tools: list[str] = cfg.get("tools") or []

    total = 0
    with open(input_path, "w") as f:
        for task in tasks:
            user_message = task["user_message"]
            image = task.get("image")
            scenario_id = hashlib.sha256(f"{user_message}{image}".encode()).hexdigest()[:16]
            for _ in range(n):
                row = {
                    "id": str(uuid.uuid4()),
                    "scenario_id": scenario_id,
                    "provider": provider,
                    "model": model,
                    "prompt_name": prompt_name,
                    "model_settings": model_settings,
                    "tools": tools,
                    "user_message": user_message,
                    "image": image,
                    "metadata": task.get("metadata", {}),
                }
                f.write(json.dumps(row) + "\n")
                total += 1

    return input_path, total


async def run(
    input_path: Path,
    agent: Agent,
    on_row_done: Callable[[], None] | None = None,
) -> tuple[int, int]:
    output_path = config.output_path / input_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("error") is None:
                    completed.add(row["id"])

    rows = []
    with open(input_path) as f:
        for line in f:
            rows.append(json.loads(line))

    succeeded = 0
    failed = 0

    with open(output_path, "a") as f:
        for row in rows:
            if row["id"] in completed:
                if on_row_done:
                    on_row_done()
                continue

            try:
                result = await run_agent(
                    agent,
                    row["user_message"],
                    image_path=row.get("image"),
                    model_settings=row.get("model_settings"),
                )
                output_row = {"id": row["id"], "output": result.output, "messages": to_jsonable_python(result.all_messages), "usage": to_jsonable_python(result.usage), "error": None}
                succeeded += 1
            except Exception as e:
                output_row = {"id": row["id"], "output": None, "messages": [], "usage": None, "error": str(e)}
                failed += 1

            f.write(json.dumps(output_row) + "\n")
            if on_row_done:
                on_row_done()

    return succeeded, failed
