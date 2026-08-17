"""providers：openai_compat 校验失败重试、anthropic schema 清洗。"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from photo_clinic.config import Config
from photo_clinic.metadata import decode_image
from photo_clinic.providers import get_provider
from photo_clinic.providers.base import UpstreamError, image_b64, image_to_data_uri
from photo_clinic.providers.openai_compat import OpenAICompatProvider
from photo_clinic.schemas import PrecheckResult

VALID = {
    "is_ai": "not_ai",
    "ai_confidence": 10,
    "ai_evidence": [],
    "category": "landscape",
    "category_confidence": 90,
    "category_reason": "景观",
    "description": "山",
}


class FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self.n = 0

    async def create(self, **kwargs):
        self.n += 1
        content = self._contents[min(self.n, len(self._contents)) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22),
        )


def make_provider(contents: list[str]) -> OpenAICompatProvider:
    config = Config(
        provider="openai_compat",
        base_url="https://example.com/v1",
        api_key="k",
        model="m",
        max_image_mb=10,
        skills_dir=None,  # type: ignore[arg-type]
    )
    provider = OpenAICompatProvider(config)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(contents))
    )
    return provider


async def call(provider: OpenAICompatProvider, image_b64: str):
    return await provider.structured_call(
        system="s",
        image=decode_image(image_b64, 10 * 1024 * 1024),
        user_text="t",
        schema=PrecheckResult,
        max_tokens=100,
        step="precheck",
        model="m",
    )


async def test_repair_retry_on_invalid_json(jpeg_image):
    provider = make_provider(["not-json at all", json.dumps(VALID, ensure_ascii=False)])
    parsed, in_tokens, out_tokens = await call(provider, jpeg_image)
    assert parsed.is_ai == "not_ai"
    assert provider._client.chat.completions.n == 2
    assert in_tokens == 11
    assert out_tokens == 22


async def test_schema_failed_after_retries(jpeg_image):
    provider = make_provider(["bad", "also bad"])
    with pytest.raises(UpstreamError) as excinfo:
        await call(provider, jpeg_image)
    assert excinfo.value.kind == "schema_failed"
    assert provider._client.chat.completions.n == 2


class RaisingCompletions:
    async def create(self, **kwargs):
        raise RuntimeError("boom")


async def test_api_error_wrapped(jpeg_image):
    provider = make_provider(["unused"])
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=RaisingCompletions()))
    with pytest.raises(UpstreamError) as excinfo:
        await call(provider, jpeg_image)
    assert excinfo.value.kind == "api_error"
    assert "boom" in str(excinfo.value)


class NullUsageCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(VALID)))],
            usage=None,
        )


async def test_usage_none_degrades_to_zero(jpeg_image):
    provider = make_provider(["unused"])
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=NullUsageCompletions()))
    parsed, in_tokens, out_tokens = await call(provider, jpeg_image)
    assert parsed.is_ai == "not_ai"
    assert in_tokens == 0
    assert out_tokens == 0


async def test_content_none_retries_then_succeeds(jpeg_image):
    provider = make_provider([None, json.dumps(VALID, ensure_ascii=False)])
    parsed, _, _ = await call(provider, jpeg_image)
    assert parsed.is_ai == "not_ai"
    assert provider._client.chat.completions.n == 2


class RecordingCompletions:
    """记录每次 create 的 messages，验证重试时附加纠错上下文。"""

    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self.n = 0
        self.messages_per_call: list[list[dict]] = []

    async def create(self, **kwargs):
        self.n += 1
        self.messages_per_call.append(list(kwargs["messages"]))
        content = self._contents[min(self.n, len(self._contents)) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


async def test_retry_appends_error_context(jpeg_image):
    completions = RecordingCompletions(["{bad", json.dumps(VALID, ensure_ascii=False)])
    provider = make_provider(["unused"])
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    parsed, _, _ = await call(provider, jpeg_image)
    assert parsed.is_ai == "not_ai"
    assert completions.n == 2
    first, second = completions.messages_per_call
    assert len(first) == 2
    assert len(second) == 4
    assert second[2] == {"role": "assistant", "content": "{bad"}
    assert "JSON" in second[3]["content"]


def test_missing_api_key_raises():
    config = Config(
        provider="openai_compat",
        base_url=None,
        api_key=None,
        model="m",
        max_image_mb=10,
        skills_dir=None,  # type: ignore[arg-type]
    )
    with pytest.raises(UpstreamError) as excinfo:
        OpenAICompatProvider(config)
    assert excinfo.value.kind == "api_error"


def make_raw_config(provider: str) -> Config:
    return Config(
        provider=provider,
        base_url="https://example.com/v1" if provider != "anthropic" else None,
        api_key="k",
        model="m",
        max_image_mb=10,
        skills_dir=None,  # type: ignore[arg-type]
    )


def test_get_provider_openai_compat():
    assert isinstance(get_provider(make_raw_config("openai_compat")), OpenAICompatProvider)


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="未知 provider"):
        get_provider(make_raw_config("bogus"))


def test_get_provider_anthropic_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # 模拟未安装
    with pytest.raises(UpstreamError) as excinfo:
        get_provider(make_raw_config("anthropic"))
    assert "anthropic SDK" in str(excinfo.value)


def test_image_b64_and_data_uri(jpeg_image):
    img = decode_image(jpeg_image, 10 * 1024 * 1024)
    assert image_b64(img) == jpeg_image
    assert image_to_data_uri(img) == f"data:image/jpeg;base64,{jpeg_image}"


def test_anthropic_schema_sanitizer():
    from photo_clinic.providers.anthropic_provider import _to_api_schema

    schema = _to_api_schema(PrecheckResult)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "title" not in schema

    # Anthropic 要求所有 object 节点带 additionalProperties:false（含 $defs 中的引用目标）
    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
