import os

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from zenml.client import Client
from zenml.exceptions import EntityExistsError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Required settings even when working locally. ---

    # OpenAI API
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    # Huggingface API
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # Comet ML (during training)
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "twin"

    # --- Required settings when deploying the code. ---
    # --- Otherwise, default values values work fine. ---

    # MongoDB database
    DATABASE_HOST: str = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "twin"

    # Qdrant vector database
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # AWS Authentication
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None

    # --- Optional settings used to tweak the code. ---

    # AWS SageMaker
    HF_MODEL_ID: str = "mlabonne/TwinLlama-3.1-8B-DPO"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    COPIES: int = 1  # Number of replicas
    GPUS: int = 1  # Number of GPUs
    CPUS: int = 2  # Number of CPU cores

    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "twin"
    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin"
    TEMPERATURE_INFERENCE: float = 0.01
    TOP_P_INFERENCE: float = 0.9
    MAX_NEW_TOKENS_INFERENCE: int = 150

    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"
    RAG_USE_QUERY_EXPANSION: bool = False
    RAG_USE_SELF_QUERY: bool = False
    RAG_USE_RERANKING: bool = True

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    # Local-first runtime
    USE_CLOUD: bool = False
    USE_ZENML_SECRET_STORE: bool = False
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_CHAT_MODEL: str = "qwen2.5:7b-instruct"
    LOCAL_MODEL_PATH: str | None = None
    LOCAL_LLM_TIMEOUT_SECONDS: int = 120
    LOCAL_LLM_MAX_RETRIES: int = 2
    USE_OPIK: bool = False
    USE_HUGGINGFACE_HUB: bool = False
    USE_MLFLOW: bool = True
    MLFLOW_TRACKING_URI: str = "file:data/mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "local-llm-twin"

    USE_LOCAL_MODELS: bool = True

    # Local training defaults for RTX 3060 12GB. These are intentionally conservative.
    LOCAL_TRAINING_BASE_MODEL: str = "models/mistral-7b"
    LOCAL_TRAINING_OUTPUT_DIR: str = "data/training/runs/mistral-7b-local-qlora"
    LOCAL_TRAINING_DATA_DIR: str = "data/training/datasets"
    LOCAL_TRAINING_MAX_SEQ_LENGTH: int = 1024
    LOCAL_TRAINING_LOAD_IN_4BIT: bool = True
    LOCAL_TRAINING_LORA_RANK: int = 8
    LOCAL_TRAINING_LORA_ALPHA: int = 16
    LOCAL_TRAINING_LORA_DROPOUT: float = 0.05
    LOCAL_TRAINING_BATCH_SIZE: int = 1
    LOCAL_TRAINING_GRADIENT_ACCUMULATION_STEPS: int = 16
    LOCAL_TRAINING_LEARNING_RATE: float = 2e-4
    LOCAL_TRAINING_NUM_EPOCHS: int = 1

    # Backwards-compatible property names for earlier local experiments.
    @property
    def use_cloud(self) -> bool:
        return self.USE_CLOUD

    @property
    def use_local_models(self) -> bool:
        return self.USE_LOCAL_MODELS

    @property
    def local_model_path(self) -> str | None:
        return self.LOCAL_MODEL_PATH

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)

        return max_token_window

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store. If the secret does not exist, it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """

        if os.getenv("USE_ZENML_SECRET_STORE", "false").lower() != "true":
            return Settings()

        try:
            logger.info("Loading settings from the ZenML secret store.")

            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError, OSError):
            logger.warning(
                "Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file."
            )
            settings = Settings()

        return settings

    def export(self) -> None:
        """
        Exports the settings to the ZenML secret store.
        """

        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning(
                "Secret 'scope' already exists. Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )


settings = Settings.load_settings()
