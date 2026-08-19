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
from photo_clinic.pipeline import (
    MAJOR_DIMENSION_CAP,
    PIXEL_SKIN_MAJOR_MESSAGE,
    _apply_major_score_cap,
    _apply_skin_rule,
    _strip_texture_major,
    run_review,
)
from photo_clinic.schemas import (
    BranchReviewResult,
    DimensionScore,
    ReviewRequest,
    SkinReport,
    TextureReport,
)


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


REVIEW_POSSIBLE = {**REVIEW, "possible_issues": ["肤色"]}
REVIEW_CLEAN = {**REVIEW, "dimensions": [{"dimension": "构图", "score": 8.0, "comment": "尚可"}]}
REVIEW_WITH_MAJOR = {
    **REVIEW_CLEAN,
    "total_score": 6.0,
    "major_issues": ["人物肤色发白发青，色温偏差导致偏离真实人体肤色"],
}


async def test_portrait_recheck_triggered_by_possible_issues_and_catches(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 首轮自评肤色不确定 → 触发复核，复核轮补判 → 按更严格结论（复核轮）整体替换
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script(
        "review", [REVIEW_POSSIBLE, REVIEW_WITH_MAJOR]
    )
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    rev_calls = [c for c in fake_provider.calls if c["step"] == "review"]
    assert len(rev_calls) == 2
    assert "第一轮认为以下方面可能存在重大问题" in rev_calls[1]["system"]
    assert "肤色" in rev_calls[1]["system"]
    assert resp.review.major_issues == REVIEW_WITH_MAJOR["major_issues"]
    assert resp.review.total_score == 6.0
    assert resp.usage.input_tokens == 300  # 预检 + 两轮评审


async def test_portrait_recheck_agreement_keeps_first(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 触发复核但复核轮也无重大问题 → 沿用第一轮
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script(
        "review", [REVIEW_POSSIBLE, REVIEW_POSSIBLE]
    )
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    rev_calls = [c for c in fake_provider.calls if c["step"] == "review"]
    assert len(rev_calls) == 2
    assert resp.review.major_issues == []
    assert resp.review.total_score == 7.5


async def test_portrait_recheck_skipped_when_clean_first_round(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 首轮无扣分点且无不确定项 → 不再复核，省一次调用
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script("review", REVIEW_CLEAN)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    rev_calls = [c for c in fake_provider.calls if c["step"] == "review"]
    assert len(rev_calls) == 1
    assert resp.review.major_issues == []
    assert resp.usage.input_tokens == 200  # 预检 + 单轮评审


async def test_portrait_recheck_triggered_by_deductions_alone(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 首轮无 uncertain 标注但有扣分点 → 也触发复核（折中触发条件）
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script(
        "review", [REVIEW, REVIEW_WITH_MAJOR]
    )
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    rev_calls = [c for c in fake_provider.calls if c["step"] == "review"]
    assert len(rev_calls) == 2
    assert "第一轮扣分点" in rev_calls[1]["system"]
    assert "地平线歪斜" in rev_calls[1]["system"]
    assert resp.review.major_issues == REVIEW_WITH_MAJOR["major_issues"]


async def test_portrait_recheck_skipped_when_confident_with_major_found(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 首轮已判出重大问题且无不确定项 → 不再复核
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script("review", REVIEW_WITH_MAJOR)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    rev_calls = [c for c in fake_provider.calls if c["step"] == "review"]
    assert len(rev_calls) == 1
    assert resp.review.major_issues == REVIEW_WITH_MAJOR["major_issues"]


async def test_portrait_recheck_runs_once_and_not_recursive(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 复核轮输出也带 possible_issues → 仍只跑一轮复核，不递归
    recheck_payload = {**REVIEW_WITH_MAJOR, "possible_issues": ["皮肤质感"]}
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script(
        "review", [REVIEW_POSSIBLE, recheck_payload]
    )
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    rev_calls = [c for c in fake_provider.calls if c["step"] == "review"]
    assert len(rev_calls) == 2
    assert resp.review.major_issues == recheck_payload["major_issues"]


async def test_non_portrait_review_skips_recheck(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 重大问题语义仅人物板块：landscape 即使带 possible_issues 也不走复核
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW_POSSIBLE)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    rev_calls = [c for c in fake_provider.calls if c["step"] == "review"]
    assert len(rev_calls) == 1
    assert resp.route == "landscape"


async def test_portrait_pixel_skin_rule_catches_when_model_reports_natural(
    skills, tmp_path, fake_provider
):
    # 模型报告 natural 时，像素检测兜底判发白 → 重大问题（确定性，不依赖模型感知）
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 400), (230, 228, 224)).save(buf, format="JPEG")  # 青白/灰白皮肤
    pale_image = base64.b64encode(buf.getvalue()).decode()
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script("review", REVIEW_CLEAN)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=pale_image))
    assert resp.route == "portrait"
    assert PIXEL_SKIN_MAJOR_MESSAGE in resp.review.major_issues


async def test_portrait_pixel_skin_rule_skips_warm_skin(
    skills, tmp_path, fake_provider
):
    # 暖调肤色（高饱和）→ 像素检测不触发
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 400), (205, 155, 115)).save(buf, format="JPEG")
    warm_image = base64.b64encode(buf.getvalue()).decode()
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script("review", REVIEW_CLEAN)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=warm_image))
    assert resp.route == "portrait"
    assert resp.review.major_issues == []


def make_result(**kw) -> BranchReviewResult:
    return BranchReviewResult(total_score=7.0, dimensions=[], **kw)


def test_strip_texture_major_removes_retouch_keeps_others():
    result = make_result(
        major_issues=[
            "皮肤质感处理过度，毛孔不可见，塑料感明显",
            "肤色在冷光环境下偏青白，属异常处理",
        ]
    )
    updated = _strip_texture_major(result)
    assert updated.major_issues == ["肤色在冷光环境下偏青白，属异常处理"]


def test_strip_texture_major_keeps_unrelated_issues():
    result = make_result(major_issues=["曝光硬伤：窗外大面积过曝死白"])
    assert _strip_texture_major(result) is result


def test_skin_rule_injects_major_when_pale_or_white():
    for whiteness in ("pale", "white"):
        result = make_result(skin_report=SkinReport(whiteness=whiteness, exempt=False))
        updated = _apply_skin_rule(result)
        assert len(updated.major_issues) == 1
        assert "肤色异常" in updated.major_issues[0]


def test_skin_rule_skips_natural_or_exempt():
    for rep in (
        SkinReport(whiteness="natural", exempt=False),
        SkinReport(whiteness="white", exempt=True),  # 非人类色/强风格化豁免
    ):
        updated = _apply_skin_rule(make_result(skin_report=rep))
        assert updated.major_issues == []


def test_skin_rule_no_duplicate_when_model_already_flagged():
    result = make_result(
        major_issues=["肤色在冷光环境下偏青白，属异常处理"],
        skin_report=SkinReport(whiteness="pale", exempt=False),
    )
    updated = _apply_skin_rule(result)
    assert updated.major_issues == ["肤色在冷光环境下偏青白，属异常处理"]


def scored_result(total: float, *dims) -> BranchReviewResult:
    return BranchReviewResult(total_score=total, dimensions=list(dims))


def test_major_score_cap_caps_post_processing_and_recomputes_total():
    result = scored_result(
        9.0,
        DimensionScore(dimension="构图", score=3.0, comment="x"),
        DimensionScore(dimension="光线", score=4.0, comment="x"),
        DimensionScore(dimension="后期", score=2.0, comment="x"),
    )
    result.major_issues = ["肤色在冷光环境下偏青白，属异常处理"]
    updated = _apply_major_score_cap(result)
    late = next(d for d in updated.dimensions if d.dimension == "后期")
    assert late.score == MAJOR_DIMENSION_CAP
    assert late.original_score == 2.0  # 原始小计保留，供展示「常规 X − 重大扣 Y = 实得 Z」
    assert late.major_deduction == 1.0
    assert updated.total_score == 3.0 + 4.0 + MAJOR_DIMENSION_CAP


def test_major_score_cap_maps_exposure_to_lighting():
    result = scored_result(
        7.0,
        DimensionScore(dimension="构图", score=3.0, comment="x"),
        DimensionScore(dimension="光线", score=2.0, comment="x"),
        DimensionScore(dimension="后期", score=2.0, comment="x"),
    )
    result.major_issues = ["曝光硬伤：窗外大面积过曝死白"]
    updated = _apply_major_score_cap(result)
    light = next(d for d in updated.dimensions if d.dimension == "光线")
    assert light.score == MAJOR_DIMENSION_CAP


def test_major_score_cap_noop_without_major_issues():
    result = scored_result(
        9.0,
        DimensionScore(dimension="构图", score=3.0, comment="x"),
        DimensionScore(dimension="光线", score=4.0, comment="x"),
        DimensionScore(dimension="后期", score=2.0, comment="x"),
    )
    updated = _apply_major_score_cap(result)
    assert updated.total_score == 9.0
    assert all(d.score == s for d, s in zip(updated.dimensions, (3.0, 4.0, 2.0)))


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


async def test_suspect_ai_warns_above_threshold(skills, jpeg_image, tmp_path, fake_provider):
    # 置信度 ≥50% 的 uncertain 才有疑似 AI 点评
    fake_provider.script("precheck", PRECHECK_UNCERTAIN_LANDSCAPE).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "landscape"
    assert resp.ai_suspicion is not None
    assert "疑似 AI 生成" in resp.ai_suspicion.warning
    system = fake_provider.calls[1]["system"]
    assert "landscape-review" in system
    # 不再追加 AI rubric / 提示词要求：疑似路径也严格按本板块契约输出
    assert "不要使用其他评分体系" not in system
    assert "prompt_suggestion" not in system


async def test_uncertain_below_threshold_no_warning(skills, jpeg_image, tmp_path, fake_provider):
    # 置信度 <50% 的 uncertain 不提示疑似 AI（视为普通照片）
    low = {**PRECHECK_UNCERTAIN_LANDSCAPE, "ai_confidence": 40}
    fake_provider.script("precheck", low).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert resp.route == "landscape"
    assert resp.ai_suspicion is None


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
    # 两轮分歧 → uncertain，置信度取均值 (60+10)/2 = 35% < 50% → 不提示疑似 AI
    assert resp.route == "landscape"
    assert resp.ai_detection.verdict == "uncertain"
    assert resp.ai_detection.confidence == pytest.approx(0.35)
    assert resp.ai_suspicion is None
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
