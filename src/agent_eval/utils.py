import json
from pathlib import Path

from pydantic_core import to_jsonable_python

from agent_eval.schemas import InputRow


def save_json(data: object, path: str | Path) -> None:
    """Utility function to save data as JSON."""
    with open(path, "w") as f:
        json.dump(to_jsonable_python(data), f, indent=2)


def read_first_row(path: str | Path) -> InputRow:
    with open(path) as f:
        return InputRow.model_validate_json(f.readline())
