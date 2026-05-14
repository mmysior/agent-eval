import importlib
from collections.abc import Callable

from pydantic_ai import Tool


def _resolve(short_path: str) -> Callable:
    module_path, func_name = short_path.rsplit(".", 1)
    try:
        module = importlib.import_module(f"{__package__}.{module_path}")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Tool module not found: '{module_path}' (from '{short_path}')")
    try:
        return getattr(module, func_name)
    except AttributeError:
        available = [n for n in dir(module) if not n.startswith("_") and callable(getattr(module, n)) and getattr(getattr(module, n), "__module__", None) == module.__name__]
        raise AttributeError(f"Tool '{func_name}' not found in '{module_path}'. Available: {available}")


def load_tools(short_paths: list[str]) -> tuple[Callable, ...]:
    return tuple(_resolve(p) for p in short_paths)


def extract_definitions(short_paths: list[str]) -> dict:
    definitions = {}
    for path in short_paths:
        tool = Tool(_resolve(path)).tool_def
        definitions[tool.name] = {
            "description": tool.description,
            "schema": tool.parameters_json_schema,
        }
    return definitions
