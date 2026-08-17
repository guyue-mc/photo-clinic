"""pipeline.py：4 条路由 + 元数据短路 + 疑似 AI 路径 + 调用计数。"""
from __future__ import annotations

import pytest
from conftest import (
    PRECHECK_AI,
    PRECHECK_LANDSCAPE,
    PRECHECK_OTHER,
    PRECHECK_PORTRAIT,
    PRECHECK_UNCERTAIN_LANDSCAPE,
    REVIEW,
)

from photo_clinic.config import Config
from photo_clinic.metadata import InvalidMediaTypeError
from photo_clinic.pipeline import run_review
from photo_clinic.schemas import ReviewRequest


def make_config(tmp_path) -> Config:
    return Config(
        provider="openai_compat",
        base_url=None,
        api_key="test-key",
        model="test-model",
        max_image_mb=10,
        skills_dir=tmp_path,
    )


async def run(skills, tmp_path, fake_provider, request: ReviewRequest):
    return await run_review(request, config=make_config(tmp_path), skills=skills, llm=fake_provider)


async def test_route_ai_via_llm(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_AI).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "ai"
    assert resp.subject is None
    assert resp.review.skill == "ai-image-review"
    assert resp.review.prompt_suggestion
    assert resp.ai_suspicion is None
    assert len(fake_provider.calls) == 2
    assert "ai-image-review" in fake_provider.calls[1]["system"]


async def test_route_landscape(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "landscape"
    assert resp.subject.category == "landscape"
    assert resp.review.skill == "landscape-review"
    assert resp.ai_suspicion is None
    assert len(fake_provider.calls) == 2
    assert "landscape-review" in fake_provider.calls[1]["system"]
    assert "ai-image-review" not in fake_provider.calls[1]["system"]


async def test_route_portrait(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "portrait"
    assert resp.review.skill == "portrait-review"
    assert resp.ai_suspicion is None


async def test_route_rejected_makes_single_call(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_OTHER)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "rejected"
    assert resp.review is None
    assert resp.subject.category == "other"
    assert len(fake_provider.calls) == 1


async def test_metadata_short_circuit_skips_precheck(
    skills, png_with_parameters, tmp_path, fake_provider
):
    fake_provider.script("review", REVIEW)
    resp = await run(
        skills, tmp_path, fake_provider, ReviewRequest(image_base64=png_with_parameters)
    )
    assert resp.route == "ai"
    assert resp.ai_detection.source == "metadata"
    assert resp.ai_detection.confidence == 0.99
    assert resp.ai_detection.metadata.has_parameters_chunk is True
    assert len(fake_provider.calls) == 1
    assert fake_provider.calls[0]["step"] == "review"


async def test_suspect_ai_appends_ai_rubric_and_warns(
    skills, jpeg_image, tmp_path, fake_provider
):
    fake_provider.script("precheck", PRECHECK_UNCERTAIN_LANDSCAPE).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "landscape"
    assert resp.ai_suspicion is not None
    assert "疑似 AI 生成" in resp.ai_suspicion.warning
    system = fake_provider.calls[1]["system"]
    assert "landscape-review" in system
    assert "不要使用其他评分体系" in system  # 防止拼接 AI rubric 导致评分维度冲突
    assert resp.review.prompt_suggestion


async def test_media_type_mismatch_raises(skills, jpeg_image, tmp_path, fake_provider):
    with pytest.raises(InvalidMediaTypeError):
        await run(
            skills,
            tmp_path,
            fake_provider,
            ReviewRequest(image_base64=jpeg_image, media_type="image/png"),
        )
    assert fake_provider.calls == []


async def test_request_model_overrides_config(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    resp = await run(
        skills,
        tmp_path,
        fake_provider,
        ReviewRequest(image_base64=jpeg_image, model="custom-model"),
    )
    assert resp.model == "custom-model"
    assert [c["model"] for c in fake_provider.calls] == ["custom-model", "custom-model"]


async def test_usage_accumulated_across_calls(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.usage.input_tokens == 200
    assert resp.usage.output_tokens == 100


async def test_usage_rejected_single_call(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_OTHER)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 50


async def test_usage_metadata_short_circuit(skills, png_with_parameters, tmp_path, fake_provider):
    fake_provider.script("review", REVIEW)
    resp = await run(
        skills, tmp_path, fake_provider, ReviewRequest(image_base64=png_with_parameters)
    )
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 50


async def test_ai_reason_joins_multiple_evidence(skills, jpeg_image, tmp_path, fake_provider):
    pre = {
        **PRECHECK_AI,
        "ai_evidence": [
            {"aspect": "hand_anatomy", "finding": "六根手指", "supports_ai": True},
            {"aspect": "lighting", "finding": "光效不自然", "supports_ai": True},
        ],
    }
    fake_provider.script("precheck", pre).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "ai"
    assert "hand_anatomy" in resp.ai_detection.reason
    assert "lighting" in resp.ai_detection.reason
    assert "；" in resp.ai_detection.reason


async def test_ai_reason_falls_back_to_description(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", {**PRECHECK_AI, "ai_evidence": []}).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.ai_detection.reason == "一张人像"


async def test_not_ai_reason_text(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.ai_detection.reason == "未发现典型 AI 生成特征"


async def test_uncertain_without_evidence_warning(skills, jpeg_image, tmp_path, fake_provider):
    pre = {**PRECHECK_UNCERTAIN_LANDSCAPE, "ai_evidence": []}
    fake_provider.script("precheck", pre).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert "证据不足，无法确定" in resp.ai_suspicion.warning


async def test_ai_confidence_clamped_in_response(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", {**PRECHECK_AI, "ai_confidence": 150}).script(
        "review", REVIEW
    )
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.ai_detection.confidence == 1.0
    assert resp.route_confidence == 1.0


async def test_borderline_uncertain_triggers_recheck_and_disagreement_is_uncertain(
    skills, jpeg_image, tmp_path, fake_provider
):
    first = {**PRECHECK_UNCERTAIN_LANDSCAPE, "ai_confidence": 60}  # 落在 [30,70] 区间
    second = {**PRECHECK_LANDSCAPE, "ai_confidence": 10}  # 复核翻为 not_ai
    fake_provider.script("precheck", [first, second]).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    pre_calls = [c for c in fake_provider.calls if c["step"] == "precheck"]
    assert len(pre_calls) == 2
    assert "第一轮证据" in pre_calls[1]["system"]
    # 两轮分歧 → uncertain，置信度取均值 (60+10)/2
    assert resp.route == "landscape"
    assert resp.ai_detection.verdict == "uncertain"
    assert resp.ai_detection.confidence == pytest.approx(0.35)
    assert resp.ai_suspicion is not None
    assert resp.usage.input_tokens == 300  # 两次预检 + 一次评审


async def test_low_confident_other_reclassifies(skills, jpeg_image, tmp_path, fake_provider):
    first = {**PRECHECK_OTHER, "category_confidence": 50}  # 低置信 other → 触发分类复核
    second = PRECHECK_LANDSCAPE  # 复核归入 landscape
    fake_provider.script("precheck", [first, second]).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    pre_calls = [c for c in fake_provider.calls if c["step"] == "precheck"]
    assert len(pre_calls) == 2
    assert "上次分类结果为 other" in pre_calls[1]["system"]
    assert resp.route == "landscape"
    assert resp.subject.category == "landscape"


async def test_confident_other_rejects_without_reclassify(
    skills, jpeg_image, tmp_path, fake_provider
):
    fake_provider.script("precheck", {**PRECHECK_OTHER, "category_confidence": 95}).script(
        "review", REVIEW
    )
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert len([c for c in fake_provider.calls if c["step"] == "precheck"]) == 1
    assert resp.route == "rejected"


async def test_cosplay_chain_recheck_then_reclassify(skills, jpeg_image, tmp_path, fake_provider):
    # 模拟 cosplay 边界图：AI 复核 + 分类复核全链路
    fake_provider.script(
        "precheck",
        [
            {"is_ai": "ai", "ai_confidence": 85, "category": "other", "category_confidence": 50,
             "ai_evidence": [], "category_reason": "x", "description": "d"},
            {"is_ai": "uncertain", "ai_confidence": 60, "category": "other", "category_confidence": 50,
             "ai_evidence": [], "category_reason": "x", "description": "d"},
            {"is_ai": "uncertain", "ai_confidence": 60, "category": "portrait", "category_confidence": 95,
             "ai_evidence": [], "category_reason": "人物为主体", "description": "d"},
        ],
    ).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert len([c for c in fake_provider.calls if c["step"] == "precheck"]) == 3
    assert resp.route == "portrait"
    assert resp.ai_detection.verdict == "uncertain"
    assert resp.ai_suspicion is not None
    assert resp.review.skill == "portrait-review"


async def test_recheck_keeps_first_round_category(skills, jpeg_image, tmp_path, fake_provider):
    first = {**PRECHECK_LANDSCAPE, "is_ai": "uncertain", "ai_confidence": 60}
    second = {**PRECHECK_OTHER, "ai_confidence": 10}  # 复核轮的分类输出（other）不应生效
    fake_provider.script("precheck", [first, second]).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "landscape"  # 分类沿用第一轮
    assert resp.subject.category == "landscape"
    assert resp.ai_detection.verdict == "uncertain"


async def test_recheck_agreement_keeps_verdict(skills, jpeg_image, tmp_path, fake_provider):
    first = {**PRECHECK_AI, "ai_confidence": 60}
    second = {**PRECHECK_AI, "ai_confidence": 80}
    fake_provider.script("precheck", [first, second]).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "ai"
    assert resp.ai_detection.verdict == "ai"
    assert resp.ai_detection.confidence == pytest.approx(0.70)


async def test_confident_ai_skips_recheck(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", {**PRECHECK_AI, "ai_confidence": 98}).script(
        "review", REVIEW
    )
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert len([c for c in fake_provider.calls if c["step"] == "precheck"]) == 1
    assert resp.route == "ai"


async def test_not_ai_skips_recheck(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert len([c for c in fake_provider.calls if c["step"] == "precheck"]) == 1
    assert resp.route == "landscape"
