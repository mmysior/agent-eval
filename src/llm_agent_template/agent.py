from dataclasses import dataclass
from typing import Callable

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage, UsageLimits

from llm_agent_template.core.config import config
from llm_agent_template.core.prompts import get_prompt
from llm_agent_template.core.providers import get_model


@dataclass
class AgentResult:
    output: str
    all_messages: list[ModelMessage]
    usage: RunUsage


async def agent(
    user_message: str,
    *,
    tools: list[Tool | Callable] | None = None,
    message_history: list[ModelMessage] | None = None,
    provider: str = config.DEFAULT_PROVIDER,
    model: str = config.DEFAULT_MODEL,
    prompt_name: str | None = None,
) -> AgentResult:
    prompt = get_prompt(prompt_name) if prompt_name else get_prompt("agent")
    _agent = Agent(
        get_model(provider, model),
        instructions=prompt.compile(),
        tools=tools or [],
    )

    result = await _agent.run(
        user_message,
        message_history=message_history or [],
        usage_limits=UsageLimits(request_limit=config.TOOL_ITERATION_LIMIT),
    )

    return AgentResult(
        output=result.output,
        all_messages=result.all_messages(),
        usage=result.usage,
    )


if __name__ == "__main__":
    import asyncio

    from llm_agent_template.utils import save_json

    async def main():
        result = await agent("What is the capital of France?")
        save_json(result, "agent_result.json")

    asyncio.run(main())
