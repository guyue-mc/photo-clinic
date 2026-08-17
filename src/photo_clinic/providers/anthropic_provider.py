"""Claude provider（可选：pip install -e ".[claude]"）。

结构化输出用 output_config.format（json_schema 模式，output_format 已弃用），
thinking 按步骤显式配置（Opus 5/Sonnet 5 省略 = 默认开 adaptive，且 max_tokens
同时覆盖思考+输出）：
- 预检：disabled + effort medium（低层判定，确定性优先）
- 评审：adaptive + effort high（质量敏感）
不发送 temperature/top_p/top_k（Opus 5/Sonnet 5 收到即 400）。
stop_reason 先于 content 检查；refusal 的 fallbacks 兜底留待后续。
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from photo_clinic.config import Config
from photo_clinic.metadata import DecodedImage
from photo_clinic.providers.base import UpstreamError, image_b64

_STEP_THINKING = {"precheck": {"type": "disabled"}, "review": {"type": "adaptive"}}
_STEP_EFFORT = {"precheck": "medium", "review": "high"}


class AnthropicProvider:
    def __init__(self, config: Config) -> None:
        from anthropic import AsyncAnthropic  # 惰性导入：未装 [claude] 时此处报 ImportError

        self._client = AsyncAnthropic()  # 凭据走 ANTHROPIC_API_KEY / ant 登录态
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
        try:
            resp = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                thinking=_STEP_THINKING.get(step, _STEP_THINKING["review"]),
                output_config={
                    "effort": _STEP_EFFORT.get(step, "high"),
                    "format": {"type": "json_schema", "schema": _to_api_schema(schema)},
                },
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image.media_type,
                                    "data": image_b64(image),
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise UpstreamError("api_error", f"{type(exc).__name__}: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise UpstreamError("refusal", "模型拒绝处理该请求")
        if resp.stop_reason == "max_tokens":
            raise UpstreamError("truncated", "输出被截断（max_tokens 不足）")
        text = "".join(block.text for block in resp.content if block.type == "text")
        try:
            parsed = schema.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise UpstreamError("schema_failed", f"结构化输出校验失败: {exc}") from exc
        return parsed, resp.usage.input_tokens, resp.usage.output_tokens


def _to_api_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema → Anthropic json_schema：对象补 additionalProperties:false、去 title。"""

    def sanitize(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key == "title":
                    continue
                if key == "additionalProperties" and value is not False:
                    continue
                out[key] = sanitize(value)
            if node.get("type") == "object":
                out.setdefault("additionalProperties", False)
            return out
        if isinstance(node, list):
            return [sanitize(item) for item in node]
        return node

    return sanitize(model.model_json_schema())
