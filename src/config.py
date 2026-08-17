from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration for model risk and rule engine thresholds.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model Risk Thresholds
    HIGH_RISK_MODEL_THRESHOLD: float = 0.90
    MEDIUM_RISK_MODEL_THRESHOLD: float = 0.70

    # Rule Engine Thresholds
    HIGH_AMOUNT_THRESHOLD: float = 500.0
    DEVICE_UNIQUE_CARD_THRESHOLD: int = 3
    CARD_1H_VELOCITY_THRESHOLD: int = 5
    CARD_24H_VELOCITY_THRESHOLD: int = 20
    UNUSUAL_AMOUNT_RATIO_THRESHOLD: float = 3.0


settings = Settings()
