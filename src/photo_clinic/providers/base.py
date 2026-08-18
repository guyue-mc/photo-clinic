"""Provider 协议与公共错误。"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from photo_clinic.metadata import DecodedImage


class UpstreamError(Exception):
    """LLM 上游错误（api_error / refusal / truncated / schema_failed）。"""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class Provider(Protocol):
    async def structured_call(
        self,
        *,
        system: str,
        image: DecodedImage,
        user_text: str,
        schema: type[BaseModel],
        max_tokens: int,
        step: str,
        model: str,
    ) -> tuple[BaseModel, int, int]:
        """返回 (解析后的模型实例, input_tokens, output_tokens)；失败抛 UpstreamError。"""
        ...


def image_to_data_uri(image: DecodedImage) -> str:
    return f"data:{image.media_type};base64,{image_b64(image)}"


def image_b64(image: DecodedImage) -> str:
    return image.b64  # 惰性缓存：同一次请求内重复调用不再重复编码
