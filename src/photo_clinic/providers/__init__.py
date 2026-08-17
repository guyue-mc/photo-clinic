"""按配置返回 Provider 实例。"""
from __future__ import annotations

from photo_clinic.config import Config
from photo_clinic.providers.base import Provider, UpstreamError

__all__ = ["Provider", "UpstreamError", "get_provider"]


def get_provider(config: Config) -> Provider:
    if config.provider == "openai_compat":
        from photo_clinic.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(config)
    if config.provider == "anthropic":
        try:
            from photo_clinic.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(config)
        except ImportError as exc:
            raise UpstreamError(
                "api_error", "未安装 anthropic SDK：pip install -e '.[claude]'"
            ) from exc
    raise ValueError(f"未知 provider: {config.provider}")
