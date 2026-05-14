import hashlib
import logging
from pathlib import Path

import yaml
from pydantic_ai import Agent
from tqdm import tqdm

from agent_eval.agent import run_agent
from agent_eval.core.config import config
from agent_eval.schemas import InputRow, OutputRow

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
                row = InputRow(
                    id=row_id,
                    scenario_id=scenario_id,
                    **spec,
                    user_message=user_message,
                    image=image,
                    metadata=task.get("metadata") or {},
                )
                f.write(row.model_dump_json() + "\n")

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
                row = OutputRow.model_validate_json(line)
                if row.error is None:
                    completed.add(row.id)

    with open(input_path) as f:
        rows = [InputRow.model_validate_json(line) for line in f]

    succeeded = failed = 0
    pending = [row for row in rows if row.id not in completed]
    logger.info("Running %s: %d rows (%d already done)", input_path.stem, len(pending), len(completed))

    with open(output_path, "a") as f:
        for row in tqdm(pending, desc=output_path.stem, unit="row"):
            try:
                result = await run_agent(
                    agent,
                    row.user_message,
                    image_path=row.image,
                    model_settings=row.model_settings,
                )
                output_row = OutputRow(
                    id=row.id,
                    output=result.output,
                    all_messages=result.all_messages,
                    usage=result.usage,
                    error=None,
                )
                succeeded += 1
            except Exception as e:
                output_row = OutputRow(
                    id=row.id,
                    output=None,
                    all_messages=[],
                    usage=None,
                    error=str(e),
                )
                failed += 1

            f.write(output_row.model_dump_json() + "\n")

    return succeeded, failed
