from functools import wraps
from typing import Any, Callable

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from .config import config

# Type definitions
type LLModel = OpenAIChatModel | AnthropicModel | MistralModel | OpenRouterModel
type ModelFactoryFn = Callable[[str], LLModel]

model_factories: dict[str, ModelFactoryFn] = {}


def register_model_provider(name: str):
    def decorator(func: ModelFactoryFn):
        @wraps(func)
        def wrapper(model_name: str, **kwargs: Any) -> Any:
            return func(model_name, **kwargs)

        model_factories[name] = wrapper
        return wrapper

    return decorator


@register_model_provider("openai")
def get_openai_model(model_name: str, **kwargs: Any) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(api_key=config.OPENAI_API_KEY),
        settings=ModelSettings(**kwargs),
    )


@register_model_provider("anthropic")
def get_anthropic_model(model_name: str, **kwargs: Any) -> AnthropicModel:
    return AnthropicModel(
        model_name=model_name,
        provider=AnthropicProvider(api_key=config.ANTHROPIC_API_KEY),
        settings=ModelSettings(**kwargs),
    )


@register_model_provider("together")
def get_together_model(model_name: str, **kwargs: Any) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url="https://api.together.xyz/v1",
            api_key=config.TOGETHER_API_KEY,
        ),
        settings=ModelSettings(**kwargs),
    )

@register_model_provider("openrouter")
def get_openrouter_model(model_name: str, **kwargs: Any) -> OpenRouterModel:
    return OpenRouterModel(
        model_name=model_name,
        provider=OpenRouterProvider(api_key=config.OPENROUTER_API_KEY),
        settings=ModelSettings(**kwargs),
    )


@register_model_provider("ollama")
def get_ollama_model(model_name: str, **kwargs: Any) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url="http://localhost:11434/v1"),
        settings=ModelSettings(**kwargs),
    )


@register_model_provider("lmstudio")
def get_lmstudio_model(model_name: str, **kwargs: Any) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url="http://localhost:1234/v1"),
        settings=ModelSettings(**kwargs),
    )


@register_model_provider("mistral")
def get_mistral_model(model_name: str, **kwargs: Any) -> MistralModel:
    return MistralModel(
        model_name=model_name,
        provider=MistralProvider(api_key=config.MISTRAL_API_KEY),
        settings=ModelSettings(**kwargs),
    )


def get_model(
    provider: str = config.DEFAULT_PROVIDER,
    model_name: str = config.DEFAULT_MODEL,
    **kwargs: Any,
) -> LLModel:
    factory = model_factories.get(provider)
    if factory is None:
        raise ValueError(f"❌ Unsupported model provider: {provider}")
    return factory(model_name, **kwargs)
