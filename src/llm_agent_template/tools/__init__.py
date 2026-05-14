import importlib
from collections.abc import Callable


def load_tools(short_paths: list[str]) -> tuple[Callable, ...]:
    tools = []
    for path in short_paths:
        module_path, func_name = path.rsplit(".", 1)
        module = importlib.import_module(f"{__package__}.{module_path}")
        tools.append(getattr(module, func_name))
    return tuple(tools)
