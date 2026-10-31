from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Tech Truckers Cash Flow Engine"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./cash_flow.db"

    TMS_BASE_URL: str = "https://api.tms.internal"
    TMS_AUTH_TOKEN: str = "mock-token-xyz-123"

    MIN_CASH_RESERVE_THRESHOLD: float = 15000.0

    # How much of a factored invoice is held back as reserve escrow,
    # released separately ~30 days later (on top of the factoring fee).
    DEFAULT_RESERVE_ESCROW_PCT: float = 0.10
    RESERVE_ESCROW_RELEASE_DAYS: int = 30

    # Used by alerts.py to post a shortfall warning. Leave blank to just log instead.
    TEAMS_WEBHOOK_URL: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()