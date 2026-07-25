from llm_engineering.settings import Settings


def test_local_example_settings_are_local_first() -> None:
    settings = Settings(_env_file=".env.local.example")

    assert settings.USE_CLOUD is False
    assert settings.USE_ZENML_SECRET_STORE is False
    assert settings.USE_OPIK is False
    assert settings.USE_HUGGINGFACE_HUB is False
    assert settings.LLM_PROVIDER == "ollama"
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert settings.LOCAL_CHAT_MODEL == "qwen2.5:7b-instruct"
    assert settings.MLFLOW_TRACKING_URI == "file:data/mlruns"
