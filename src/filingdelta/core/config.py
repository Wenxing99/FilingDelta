from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    filingdelta_llm_model: str = Field(
        default="gpt-5.4-nano",
        alias="FILINGDELTA_LLM_MODEL",
    )
    filingdelta_embed_model: str = Field(
        default="text-embedding-3-small",
        alias="FILINGDELTA_EMBED_MODEL",
    )
    filingdelta_use_llama_parse: bool = Field(
        default=True,
        alias="FILINGDELTA_USE_LLAMA_PARSE",
    )
    filingdelta_parse_provider: Literal["local", "llama_cloud"] = Field(
        default="local",
        alias="FILINGDELTA_PARSE_PROVIDER",
    )
    filingdelta_extract_provider: Literal["structured_llm", "llama_extract"] = Field(
        default="structured_llm",
        alias="FILINGDELTA_EXTRACT_PROVIDER",
    )
    filingdelta_llama_parse_tier: str = Field(
        default="cost-effective",
        alias="FILINGDELTA_LLAMA_PARSE_TIER",
    )
    filingdelta_llama_parse_version: str = Field(
        default="latest",
        alias="FILINGDELTA_LLAMA_PARSE_VERSION",
    )
    filingdelta_llama_extract_tier: str = Field(
        default="fast",
        alias="FILINGDELTA_LLAMA_EXTRACT_TIER",
    )
    filingdelta_app_host: str = Field(
        default="127.0.0.1",
        alias="FILINGDELTA_APP_HOST",
    )
    filingdelta_app_port: int = Field(
        default=8000,
        alias="FILINGDELTA_APP_PORT",
    )
    filingdelta_qdrant_path: str = Field(
        default="./data/indexes/qdrant",
        alias="FILINGDELTA_QDRANT_PATH",
    )

    llama_cloud_api_key: str | None = Field(
        default=None,
        alias="LLAMA_CLOUD_API_KEY",
    )
    llama_cloud_base_url: str | None = Field(
        default=None,
        alias="LLAMA_CLOUD_BASE_URL",
    )

    @property
    def app_name(self) -> str:
        return "FilingDelta"

    @property
    def qdrant_path(self) -> Path:
        return (REPO_ROOT / self.filingdelta_qdrant_path).resolve()

    def require_openai_api_key(self) -> str:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for model and embedding calls.")
        return self.openai_api_key

    def require_llama_cloud_api_key(self) -> str:
        if not self.llama_cloud_api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY is required for LlamaParse and LlamaExtract calls."
            )
        return self.llama_cloud_api_key

    def llama_cloud_client_kwargs(self) -> dict[str, str]:
        kwargs = {"api_key": self.require_llama_cloud_api_key()}
        if self.llama_cloud_base_url:
            base_url = self.llama_cloud_base_url.strip()
            if not base_url.startswith(("http://", "https://")):
                base_url = f"https://{base_url}"
            kwargs["base_url"] = base_url
        return kwargs

    def safe_summary(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "llm_model": self.filingdelta_llm_model,
            "embed_model": self.filingdelta_embed_model,
            "openai_api_key_configured": bool(self.openai_api_key),
            "parse_provider": self.filingdelta_parse_provider,
            "extract_provider": self.filingdelta_extract_provider,
            "use_llama_parse": self.filingdelta_use_llama_parse,
            "llama_parse_configured": bool(self.llama_cloud_api_key),
            "llama_cloud_base_url_configured": bool(self.llama_cloud_base_url),
            "llama_parse_tier": self.filingdelta_llama_parse_tier,
            "llama_extract_tier": self.filingdelta_llama_extract_tier,
            "openai_base_url_configured": bool(self.openai_base_url),
            "qdrant_path": str(self.qdrant_path),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
