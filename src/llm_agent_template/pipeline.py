import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai import Agent
from pydantic_core import to_jsonable_python

from llm_agent_template.agent import get_agent, run_agent
from llm_agent_template.core.config import config
from llm_agent_template.tools import extract_definitions, load_tools


@dataclass
class FileStats:
    name: str
    input_path: Path
    output_path: Path
    total: int
    completed: int

    @property
    def remaining(self) -> int:
        return self.total - self.completed


def analyze(name: str) -> FileStats:
    input_path = config.input_path / name
    output_path = config.output_path / name

    with open(input_path) as f:
        total = sum(1 for _ in f)

    completed = 0
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if json.loads(line).get("error") is None:
                    completed += 1

    return FileStats(
        name=name,
        input_path=input_path,
        output_path=output_path,
        total=total,
        completed=completed,
    )


def agent_from_yaml(tasks_yaml: str | Path) -> tuple[Agent, dict]:
    with open(tasks_yaml) as f:
        cfg: dict = yaml.safe_load(f)

    tool_paths = cfg.get("tools") or []
    agent = get_agent(
        provider=cfg.get("provider", ""),
        model=cfg.get("model", ""),
        prompt_name=cfg.get("prompt_name", "agent"),
        tools=load_tools(tool_paths),
    )
    return agent, extract_definitions(tool_paths)


def prepare(tasks_yaml: str | Path) -> tuple[str, int]:
    tasks_yaml = Path(tasks_yaml)
    with open(tasks_yaml) as f:
        cfg: dict = yaml.safe_load(f)

    name = tasks_yaml.stem + ".jsonl"
    input_path = config.input_path / name
    input_path.parent.mkdir(parents=True, exist_ok=True)

    n: int = cfg.get("n", 1)
    spec = {
        "provider": cfg.get("provider", ""),
        "model": cfg.get("model", ""),
        "prompt_name": cfg.get("prompt_name", "agent"),
        "model_settings": cfg.get("model_settings") or {},
        "tools": cfg.get("tools") or [],
    }

    total = 0
    with open(input_path, "w") as f:
        for task in cfg["tasks"]:
            user_message = task["user_message"]
            image = task["image"]
            scenario_id = hashlib.sha256(f"{user_message}{image}".encode()).hexdigest()[:16]
            for _ in range(n):
                row = {
                    "id": str(uuid.uuid4()),
                    "scenario_id": scenario_id,
                    **spec,
                    "user_message": user_message,
                    "image": image,
                    "metadata": task.get("metadata", {}),
                }
                f.write(json.dumps(row) + "\n")
                total += 1

    return name, total


async def run(
    name: str,
    agent: Agent,
    tool_definitions: dict,
    on_row_done: Callable[[], Any] | None = None,
) -> tuple[int, int]:
    input_path = config.input_path / name
    output_path = config.output_path / name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("error") is None:
                    completed.add(row["id"])

    with open(input_path) as f:
        rows = [json.loads(line) for line in f]

    succeeded = failed = 0

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
                output_row = {
                    "id": row["id"],
                    "tool_definitions": tool_definitions,
                    "output": result.output,
                    "messages": to_jsonable_python(result.all_messages),
                    "usage": to_jsonable_python(result.usage),
                    "error": None,
                }
                succeeded += 1
            except Exception as e:
                output_row = {
                    "id": row["id"],
                    "tool_definitions": tool_definitions,
                    "output": None,
                    "messages": [],
                    "usage": None,
                    "error": str(e),
                }
                failed += 1

            f.write(json.dumps(output_row) + "\n")
            if on_row_done:
                on_row_done()

    return succeeded, failed
