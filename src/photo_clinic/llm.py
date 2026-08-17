"""LLM 入口：pipeline/server 只从这里拿 Provider 与错误类型。"""
from __future__ import annotations

from photo_clinic.config import Config
from photo_clinic.providers import get_provider
from photo_clinic.providers.base import Provider, UpstreamError

__all__ = ["Provider", "UpstreamError", "build_llm"]


def build_llm(config: Config) -> Provider:
    return get_provider(config)
