import hashlib
import json
import logging
from pathlib import Path

import yaml
from pydantic_ai import Agent
from pydantic_core import to_jsonable_python
from tqdm import tqdm

from agent_eval.agent import run_agent
from agent_eval.core.config import config

logger = logging.getLogger(__name__)


def prepare(tasks_yaml: str | Path) -> Path:
    tasks_yaml = Path(tasks_yaml)
    with open(tasks_yaml) as f:
        cfg: dict = yaml.safe_load(f)

    input_path = config.input_path / (tasks_yaml.stem + ".jsonl")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing %s -> %s", tasks_yaml.name, input_path)

    n: int = cfg.get("n", 1)
    spec = {
        "model": cfg.get("model", ""),
        "provider": cfg.get("provider", ""),
        "model_settings": cfg.get("model_settings") or {},
    }

    with open(input_path, "w") as f:
        for task in cfg["tasks"]:
            user_message = task["user_message"]
            image = task.get("image")
            scenario_id = hashlib.sha256(f"{user_message}{image}".encode()).hexdigest()[:16]
            for i in range(n):
                row_id = hashlib.sha256(f"{user_message}{image}{i}".encode()).hexdigest()[:16]
                row = {
                    "id": row_id,
                    "scenario_id": scenario_id,
                    **spec,
                    "user_message": user_message,
                    "image": image,
                    "metadata": task.get("metadata") or {},
                }
                f.write(json.dumps(row) + "\n")

    logger.info("Prepared %d rows -> %s", n * len(cfg["tasks"]), input_path)
    return input_path


async def run_eval(agent: Agent, input_jsonl: str | Path, output_jsonl: str | Path) -> tuple[int, int]:
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                row: dict = json.loads(line)
                if row.get("error") is None:
                    completed.add(row["id"])

    with open(input_path) as f:
        rows = [json.loads(line) for line in f]

    succeeded = failed = 0
    pending = [row for row in rows if row["id"] not in completed]
    logger.info("Running %s: %d rows (%d already done)", input_path.stem, len(pending), len(completed))

    with open(output_path, "a") as f:
        for row in tqdm(pending, desc=output_path.stem, unit="row"):
            try:
                result = await run_agent(
                    agent,
                    row["user_message"],
                    image_path=row.get("image"),
                    model_settings=row.get("model_settings"),
                )
                output_row = {
                    "id": row["id"],
                    "output": result.output,
                    "messages": to_jsonable_python(result.all_messages),
                    "usage": to_jsonable_python(result.usage),
                    "error": None,
                }
                succeeded += 1
            except Exception as e:
                output_row = {
                    "id": row["id"],
                    "output": None,
                    "messages": [],
                    "usage": None,
                    "error": str(e),
                }
                failed += 1

            f.write(json.dumps(output_row) + "\n")

    return succeeded, failed
