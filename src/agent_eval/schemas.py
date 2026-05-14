from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage


class InputRow(BaseModel):
    id: str
    scenario_id: str
    model: str
    provider: str
    model_settings: dict
    user_message: str
    image: str | None
    metadata: dict


class AgentResult(BaseModel):
    output: str
    all_messages: list[ModelMessage]
    usage: RunUsage


class OutputRow(BaseModel):
    id: str
    output: str | None
    all_messages: list[ModelMessage]
    usage: RunUsage | None
    error: str | None
