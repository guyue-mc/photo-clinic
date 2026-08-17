"""OpenAI 兼容 provider（通义/DeepSeek/智谱/豆包通用）。

结构化输出策略：response_format=json_object（国产模型最广泛支持的模式）
+ Pydantic 校验 + 失败重试一次（把校验错误附进下次请求）。
json_schema 逐级降级（json_schema → json_object → 纯 prompt）留待需要时再实现。
"""
from __future__ import annotations

import json

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from photo_clinic.config import Config
from photo_clinic.metadata import DecodedImage
from photo_clinic.providers.base import UpstreamError, image_to_data_uri

_MAX_RETRIES = 1  # 校验失败重试一次


class OpenAICompatProvider:
    def __init__(self, config: Config) -> None:
        if not config.api_key:
            raise UpstreamError("api_error", "缺少 API Key：请在 .env 设置 PHOTO_AGENT_API_KEY")
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.openai.com/v1",
        )
        self._model = config.model

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
        messages: list[dict] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(image)}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                raise UpstreamError("api_error", f"{type(exc).__name__}: {exc}") from exc
            text = resp.choices[0].message.content or ""
            try:
                parsed = schema.model_validate(json.loads(text))
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt >= _MAX_RETRIES:
                    raise UpstreamError("schema_failed", f"结构化输出校验失败: {exc}") from exc
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": f"输出不符合要求的 JSON 格式，校验错误：{exc}。"
                        "请严格按要求的 JSON 字段重新输出，只输出 JSON。",
                    }
                )
                continue
            usage = resp.usage
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            return parsed, input_tokens, output_tokens
        raise AssertionError("unreachable")
