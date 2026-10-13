from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_PATH: str = "data"

    LLM_BASE_URL: str = "http://127.0.0.1:8000/v1"
    LLM_MODEL_NAME: str = "Qwen/Qwen3-4B-Instruct-2507"

    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 256

    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()