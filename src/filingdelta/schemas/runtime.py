from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str


class RuntimeConfigResponse(BaseModel):
    app_name: str
    llm_model: str
    embed_model: str
    openai_api_key_configured: bool
    use_llama_parse: bool
    llama_parse_configured: bool
    llama_cloud_base_url_configured: bool
    llama_parse_tier: str
    llama_extract_tier: str
    openai_base_url_configured: bool
    qdrant_path: str
