from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration for model, fusion and rule thresholds."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    HIGH_RISK_MODEL_THRESHOLD: float = 0.90
    MEDIUM_RISK_MODEL_THRESHOLD: float = 0.70

    HIGH_AMOUNT_THRESHOLD: float = 500.0
    DEVICE_UNIQUE_CARD_THRESHOLD: int = 3
    CARD_1H_VELOCITY_THRESHOLD: int = 5
    CARD_24H_VELOCITY_THRESHOLD: int = 20
    UNUSUAL_AMOUNT_RATIO_THRESHOLD: float = 3.0

    FUSION_MODEL_WEIGHT: float = 0.80
    FUSION_BEHAVIORAL_WEIGHT: float = 0.20
    FUSION_REVIEW_THRESHOLD: float = 0.50
    FUSION_HIGH_THRESHOLD: float = 0.80


settings = Settings()
