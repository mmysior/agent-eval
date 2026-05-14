import logging

import click

from agent_eval.core.logging import setup_logging
from agent_eval.pipeline import prepare

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    setup_logging()


@cli.command()
@click.argument("tasks_yaml", type=click.Path(exists=True, path_type=str))
def prepare_input(tasks_yaml: str) -> None:
    """Generate input JSONL from a tasks YAML file."""
    try:
        path = prepare(tasks_yaml)
        click.echo(f"Input file ready: {path}")
    except Exception as e:
        logger.error("Failed to prepare input: %s", e)
        raise SystemExit(1)
