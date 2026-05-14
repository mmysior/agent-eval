import json
from pathlib import Path

from pydantic_core import to_jsonable_python


def save_json(data: object, path: str | Path) -> None:
    """Utility function to save data as JSON."""
    with open(path, "w") as f:
        json.dump(to_jsonable_python(data), f, indent=2)
