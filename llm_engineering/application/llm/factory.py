from llm_engineering.application.llm.providers import LLMProvider, OllamaLLMProvider
from llm_engineering.settings import settings


def get_llm_provider() -> LLMProvider:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        return OllamaLLMProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LOCAL_CHAT_MODEL,
            timeout_seconds=settings.LOCAL_LLM_TIMEOUT_SECONDS,
        )

    raise ValueError(f"Unsupported local LLM provider: {settings.LLM_PROVIDER}")
