import mimetypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from pydantic_ai import Agent, BinaryContent, Tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from agent_eval.core.config import config
from agent_eval.core.prompts import get_prompt
from agent_eval.core.providers import get_model


@dataclass
class AgentResult:
    output: str
    all_messages: list[ModelMessage]
    usage: RunUsage


@lru_cache
def get_agent(
    prompt_name: str = config.DEFAULT_PROMPT,
    provider: str = config.DEFAULT_PROVIDER,
    model: str = config.DEFAULT_MODEL,
    tools: tuple[Tool | Callable, ...] = (),
) -> Agent:
    prompt = get_prompt(prompt_name)
    return Agent(
        get_model(provider, model),
        instructions=prompt.compile(),
        tools=tools,
    )


async def run_agent(
    agent: Agent,
    user_message: str,
    image_path: str | Path | None = None,
    model_settings: dict | None = None,
    message_history: list[ModelMessage] | None = None,
) -> AgentResult:
    if image_path is not None:
        path = Path(image_path)
        media_type, _ = mimetypes.guess_type(str(path))
        prompt = [user_message, BinaryContent(data=path.read_bytes(), media_type=media_type or "image/png")]
    else:
        prompt = user_message

    result = await agent.run(
        prompt,
        message_history=message_history or [],
        usage_limits=UsageLimits(request_limit=config.TOOL_ITERATION_LIMIT),
        model_settings=ModelSettings(**(model_settings or {})),
    )
    return AgentResult(
        output=result.output,
        all_messages=result.all_messages(),
        usage=result.usage,
    )
