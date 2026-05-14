import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from llm_agent_template.agent import get_agent
from llm_agent_template.core.config import config
from llm_agent_template.core.logging import setup_logging
from llm_agent_template.pipeline import prepare, run
from llm_agent_template.tools.calc import calculate_power, calculate_transmission, convert_speed

ORANGE = "dark_orange"
GRAY = "grey50"

console = Console()


def _get_agent():
    return get_agent(
        provider=config.DEFAULT_PROVIDER,
        model=config.DEFAULT_MODEL,
        prompt_name="agent",
        tools=(convert_speed, calculate_power, calculate_transmission),
    )


@dataclass
class FileStats:
    input_path: Path
    output_path: Path
    total: int
    completed: int

    @property
    def remaining(self) -> int:
        return self.total - self.completed


def _analyze(input_path: Path) -> FileStats:
    output_path = config.output_path / input_path.name

    with open(input_path) as f:
        total = sum(1 for _ in f)

    completed = 0
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("error") is None:
                    completed += 1

    return FileStats(input_path=input_path, output_path=output_path, total=total, completed=completed)


def _print_preamble(stats: list[FileStats]) -> None:
    total_rows = sum(s.total for s in stats)
    total_remaining = sum(s.remaining for s in stats)

    console.print()
    console.print(Rule(
        f"[{ORANGE}]Agent Run[/{ORANGE}] [{GRAY}]— {len(stats)} file(s), {total_remaining}/{total_rows} generations pending[/{GRAY}]"
    ))
    console.print()

    table = Table(show_header=True, header_style=GRAY, box=None, pad_edge=False, show_edge=False)
    table.add_column("File")
    table.add_column("Input", style=GRAY)
    table.add_column("Output", style=GRAY)
    table.add_column("Total", justify="right", style=GRAY)
    table.add_column("Done", justify="right", style=GRAY)
    table.add_column("Remaining", justify="right", style=ORANGE)

    for s in stats:
        table.add_row(
            s.input_path.name,
            str(s.input_path),
            str(s.output_path),
            str(s.total),
            str(s.completed),
            str(s.remaining),
        )

    console.print(table)
    console.print()


@click.group()
def cli() -> None:
    setup_logging()


@cli.command()
@click.argument("tasks_yaml", type=click.Path(exists=True, path_type=Path))
def prepare_cmd(tasks_yaml: Path) -> None:
    """Convert a tasks YAML into an input JSONL file."""
    input_path, count = prepare(tasks_yaml)
    console.print(f"[{ORANGE}]Prepared {count} tasks →[/{ORANGE}] [{GRAY}]{input_path}[/{GRAY}]")


@cli.command()
@click.argument("file", type=click.Path(path_type=Path), required=False)
def run_cmd(file: Path | None) -> None:
    """Run generations from input JSONL files, skipping already-completed tasks.

    If FILE is omitted, all JSONL files in the input directory are processed.
    """
    if file is not None:
        input_files = [file]
    else:
        input_files = sorted(config.input_path.glob("*.jsonl"))
        if not input_files:
            console.print(f"[{GRAY}]No JSONL files found in {config.input_path}[/{GRAY}]")
            return

    all_stats = [_analyze(f) for f in input_files]
    _print_preamble(all_stats)

    agent = _get_agent()

    with Progress(
        SpinnerColumn(style=ORANGE),
        TextColumn("{task.description}"),
        BarColumn(complete_style=ORANGE, finished_style=GRAY),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for stats in all_stats:
            task_id = progress.add_task(
                f"[{GRAY}]{stats.input_path.name}[/{GRAY}]",
                total=stats.total,
            )

            succeeded, failed = asyncio.run(
                run(stats.input_path, agent, on_row_done=lambda: progress.advance(task_id))
            )

            total_ok = stats.completed + succeeded
            label = f"[{ORANGE}]{total_ok} ok[/{ORANGE}]"
            if failed:
                label += f" [{GRAY}]{failed} failed[/{GRAY}]"
            progress.update(task_id, description=f"[{GRAY}]{stats.input_path.name}[/{GRAY}] ({label})")

    console.print()
