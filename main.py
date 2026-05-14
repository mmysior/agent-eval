import asyncio
import logging
import sys

from tqdm import tqdm

from agent_eval.core.config import config
from agent_eval.core.logging import setup_logging
from agent_eval.pipeline import agent_from_yaml, analyze, prepare, run

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    yaml_files = sorted(f for f in config.tasks_path.glob("*.yaml") if not f.name.startswith("_"))
    if not yaml_files:
        logger.error("No task files found in %s", config.tasks_path)
        sys.exit(1)

    for f in yaml_files:
        input_file = config.input_path / (f.stem + ".jsonl")
        if not input_file.exists():
            name, count = prepare(f)
            logger.info("Prepared %d tasks -> %s", count, config.input_path / name)

    total_succeeded = total_failed = 0

    for f in yaml_files:
        try:
            agent, tool_defs = agent_from_yaml(f)
        except (ModuleNotFoundError, AttributeError) as e:
            logger.error("Error loading tools from %s: %s", f.name, e)
            sys.exit(1)

        stats = analyze(f.stem + ".jsonl")
        with tqdm(total=stats.total, initial=stats.completed, desc=f.stem, unit=" tasks") as bar:
            succeeded, failed = asyncio.run(run(f.stem + ".jsonl", agent, tool_defs, on_row_done=bar.update))
            bar.set_postfix(ran=succeeded, skip=stats.completed, err=failed)

        total_succeeded += succeeded
        total_failed += failed

    logger.info("Done: %d ok, %d failed", total_succeeded, total_failed)


if __name__ == "__main__":
    main()
