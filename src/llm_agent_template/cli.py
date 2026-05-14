import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from llm_agent_template.core.config import config
from llm_agent_template.core.logging import setup_logging
from llm_agent_template.pipeline import FileStats, agent_from_yaml, analyze, prepare, run

ORANGE = "dark_orange"
GRAY = "grey50"

console = Console()


def _print_preamble(stats: list[FileStats]) -> None:
    total = sum(s.total for s in stats)
    remaining = sum(s.remaining for s in stats)

    console.print()
    console.print(Rule(
        f"[{ORANGE}]Agent Run[/{ORANGE}] [{GRAY}]— {len(stats)} file(s), {remaining}/{total} generations pending[/{GRAY}]"
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
        table.add_row(s.name, str(s.input_path), str(s.output_path), str(s.total), str(s.completed), str(s.remaining))

    console.print(table)
    console.print()


@click.group()
def cli() -> None:
    setup_logging()


@cli.command()
@click.argument("tasks_yaml", type=click.Path(exists=True, path_type=Path))
def prepare_cmd(tasks_yaml: Path) -> None:
    """Convert a tasks YAML into an input JSONL file."""
    name, count = prepare(tasks_yaml)
    console.print(f"[{ORANGE}]Prepared {count} tasks →[/{ORANGE}] [{GRAY}]{config.input_path / name}[/{GRAY}]")


@cli.command()
@click.argument("tasks_yaml", type=click.Path(exists=True, path_type=Path), required=False)
def run_cmd(tasks_yaml: Path | None) -> None:
    """Run generations from a tasks YAML, skipping already-completed tasks.

    If TASKS_YAML is omitted, all non-private YAML files in the tasks directory are processed.
    """
    if tasks_yaml is None:
        yaml_files = sorted(f for f in config.tasks_path.glob("*.yaml") if not f.name.startswith("_"))
        if not yaml_files:
            console.print(f"[{GRAY}]No task files found in {config.tasks_path}[/{GRAY}]")
            return
    else:
        yaml_files = [tasks_yaml]

    missing = [f for f in yaml_files if not (config.input_path / (f.stem + ".jsonl")).exists()]
    if missing:
        names = ", ".join(f.name for f in missing)
        console.print(f"[{ORANGE}]No input file found for:[/{ORANGE}] [{GRAY}]{names}[/{GRAY}]")
        if not click.confirm("Prepare them now?", default=True):
            return
        for f in missing:
            name, count = prepare(f)
            console.print(f"[{GRAY}]Prepared {count} tasks → {config.input_path / name}[/{GRAY}]")
        console.print()

    entries = []
    for f in yaml_files:
        try:
            agent, tool_defs = agent_from_yaml(f)
        except (ModuleNotFoundError, AttributeError) as e:
            console.print(f"[{ORANGE}]Error loading tools from {f.name}:[/{ORANGE}] [{GRAY}]{e}[/{GRAY}]")
            return
        entries.append((f, agent, tool_defs, analyze(f.stem + ".jsonl")))

    _print_preamble([stats for _, _, _, stats in entries])

    with Progress(
        SpinnerColumn(style=ORANGE),
        TextColumn("{task.description}"),
        BarColumn(complete_style=ORANGE, finished_style=GRAY),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for _, agent, tool_definitions, stats in entries:
            task_id = progress.add_task(f"[{GRAY}]{stats.name}[/{GRAY}]", total=stats.total)

            succeeded, failed = asyncio.run(
                run(stats.name, agent, tool_definitions, on_row_done=lambda: progress.advance(task_id))
            )

            total_ok = stats.completed + succeeded
            label = f"[{ORANGE}]{total_ok} ok[/{ORANGE}]"
            if failed:
                label += f" [{GRAY}]{failed} failed[/{GRAY}]"
            progress.update(task_id, description=f"[{GRAY}]{stats.name}[/{GRAY}] ({label})")

    console.print()
