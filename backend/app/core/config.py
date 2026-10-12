"""Application configuration settings using pydantic-settings."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings schema and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Vector Workbench"
    version: str = "1.0.0"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    demo_mode: bool = True

    # Database
    database_url: str = "sqlite:///./vector_workbench.sqlite"

    # Keycloak OIDC Authentication
    keycloak_url: str = "http://127.0.0.1:8080"
    keycloak_realm: str = "vector-workbench"
    keycloak_client_id: str = "vector-workbench-app"
    keycloak_client_secret: str = ""
    jwt_secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"

    # Opt-in LLM Configuration (Only LLM_API_KEY is secret)
    llm_api_key: str = ""
    llm_provider: str = "openai-compatible"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""

    def is_production(self) -> bool:
        """Check if environment is set to production."""
        return self.environment.lower() in ("production", "prod")

    def ensure_safe_mode(self) -> None:
        """Enforce production safety: refuse local demo mode when in production."""
        if self.is_production() and self.demo_mode:
            raise RuntimeError(
                "Fatal configuration error: demo_mode cannot be active in production mode."
            )


settings = Settings()
