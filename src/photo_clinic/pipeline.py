"""评审编排：预检 → 路由 → 板块评审（含疑似 AI 路径）。"""
from __future__ import annotations

from photo_clinic.config import Config
from photo_clinic.metadata import (
    DecodedImage,
    InvalidMediaTypeError,
    decode_image,
    inspect_metadata,
)
from photo_clinic.prompts import (
    PRECHECK_RECHECK_USER_TEXT,
    PRECHECK_RECLASSIFY_USER_TEXT,
    PRECHECK_USER_TEXT,
    REVIEW_USER_TEXT,
    build_precheck_system,
    build_recheck_system,
    build_reclassify_system,
    build_review_system,
)
from photo_clinic.providers.base import Provider
from photo_clinic.registry import ROUTE_TO_SKILL
from photo_clinic.schemas import (
    AiDetection,
    AiSuspicion,
    BranchReview,
    BranchReviewResult,
    PrecheckResult,
    ReviewRequest,
    ReviewResponse,
    SubjectClassification,
    Usage,
)
from photo_clinic.skills import SkillRegistry

PRECHECK_MAX_TOKENS = 2048
REVIEW_MAX_TOKENS = 4096
METADATA_AI_CONFIDENCE = 0.99
# 临界复核：置信度落在 [30,70] 区间，或判 ai 但置信不足 95% 时，
# 带第一轮证据再复核一次（抑制边界样本上单次判定的随机翻转）
RECHECK_CONF_LOW, RECHECK_CONF_HIGH = 30.0, 70.0
RECHECK_AI_CONF_MAX = 95.0
# 分类复核：other 但置信不足 90% 时强制二选一复核（拒绝是硬动作，误拒比误收代价大）
CLASSIFY_OTHER_MIN_CONF = 90.0


async def run_review(
    request: ReviewRequest,
    *,
    config: Config,
    skills: SkillRegistry,
    llm: Provider,
) -> ReviewResponse:
    usage = Usage()
    # 超上限的图片自动压缩（长边 ≤2000px 且 ≤max_bytes）后再评审
    image = decode_image(request.image_base64, config.max_image_bytes, auto_compress=True)
    if request.media_type and request.media_type != image.media_type:
        raise InvalidMediaTypeError(
            f"声明的 media_type（{request.media_type}）与实际（{image.media_type}）不符"
        )
    model = request.model or config.model

    flags = inspect_metadata(image)
    if flags.evidence_strength == "strong":
        # 元数据实锤 → 短路判 AI，跳过 LLM 预检，直接 AI 板块评审
        ai_detection = AiDetection(
            verdict="ai",
            confidence=METADATA_AI_CONFIDENCE,
            reason="图片元数据包含 AI 生成证据（C2PA 签名 / PNG 参数块 / 生成软件字段）",
            metadata=flags,
            source="metadata",
        )
        review, tokens = await _review_branch(llm, skills, image, "ai", suspect_ai=False, model=model)
        _add_usage(usage, tokens)
        return ReviewResponse(
            route="ai",
            route_confidence=METADATA_AI_CONFIDENCE,
            ai_detection=ai_detection,
            review=review,
            model=model,
            usage=usage,
        )

    pre, in_tokens, out_tokens = await llm.structured_call(
        system=build_precheck_system(skills, flags),
        image=image,
        user_text=PRECHECK_USER_TEXT,
        schema=PrecheckResult,
        max_tokens=PRECHECK_MAX_TOKENS,
        step="precheck",
        model=model,
    )
    _add_usage(usage, (in_tokens, out_tokens))

    if _needs_recheck(pre):
        pre, recheck_tokens = await _recheck_precheck(llm, skills, image, flags, pre, model)
        _add_usage(usage, recheck_tokens)

    ai_confidence = pre.ai_confidence / 100.0
    ai_reason = _ai_reason(pre)
    ai_detection = AiDetection(
        verdict=pre.is_ai,
        confidence=ai_confidence,
        reason=ai_reason,
        checks=pre.ai_evidence,
        metadata=flags,
        source="llm",
    )
    if pre.is_ai == "ai":
        review, tokens = await _review_branch(llm, skills, image, "ai", suspect_ai=False, model=model)
        _add_usage(usage, tokens)
        return ReviewResponse(
            route="ai",
            route_confidence=ai_confidence,
            ai_detection=ai_detection,
            review=review,
            model=model,
            usage=usage,
        )

    if pre.category == "other" and pre.category_confidence < CLASSIFY_OTHER_MIN_CONF:
        # 低置信的 other 直接拒绝太激进：强制风景/人物二选一复核，避免误拒
        pre, reclassify_tokens = await _reclassify(llm, skills, image, flags, pre, model)
        _add_usage(usage, reclassify_tokens)

    category_confidence = pre.category_confidence / 100.0
    subject = SubjectClassification(
        category=pre.category,
        confidence=category_confidence,
        reason=pre.category_reason,
    )
    if pre.category == "other":
        # 严格两类：其他拒绝，不做评审调用
        return ReviewResponse(
            route="rejected",
            route_confidence=category_confidence,
            ai_detection=ai_detection,
            subject=subject,
            model=model,
            usage=usage,
        )

    suspect_ai = pre.is_ai == "uncertain"
    review, tokens = await _review_branch(
        llm, skills, image, pre.category, suspect_ai=suspect_ai, model=model
    )
    _add_usage(usage, tokens)
    ai_suspicion = None
    if suspect_ai:
        ai_suspicion = AiSuspicion(
            warning=f"疑似 AI 生成（置信度 {pre.ai_confidence:.0f}%）：{ai_reason}"
        )
    return ReviewResponse(
        route=pre.category,  # 此处 category 已排除 other，必为 landscape/portrait
        route_confidence=category_confidence,
        ai_detection=ai_detection,
        subject=subject,
        review=review,
        ai_suspicion=ai_suspicion,
        model=model,
        usage=usage,
    )


async def _review_branch(
    llm: Provider,
    skills: SkillRegistry,
    image: DecodedImage,
    route: str,
    *,
    suspect_ai: bool,
    model: str,
) -> tuple[BranchReview, tuple[int, int]]:
    result, in_tokens, out_tokens = await llm.structured_call(
        system=build_review_system(skills, route, suspect_ai=suspect_ai),
        image=image,
        user_text=REVIEW_USER_TEXT,
        schema=BranchReviewResult,
        max_tokens=REVIEW_MAX_TOKENS,
        step="review",
        model=model,
    )
    review = BranchReview(
        skill=ROUTE_TO_SKILL[route],
        total_score=result.total_score,
        dimensions=result.dimensions,
        improvements=result.improvements,
        bonus_notes=result.bonus_notes,
        prompt_suggestion=result.prompt_suggestion,
    )
    return review, (in_tokens, out_tokens)


def _add_usage(usage: Usage, tokens: tuple[int, int]) -> None:
    usage.input_tokens += tokens[0]
    usage.output_tokens += tokens[1]


def _needs_recheck(pre: PrecheckResult) -> bool:
    if RECHECK_CONF_LOW <= pre.ai_confidence <= RECHECK_CONF_HIGH:
        return True
    return pre.is_ai == "ai" and pre.ai_confidence < RECHECK_AI_CONF_MAX


async def _recheck_precheck(
    llm: Provider,
    skills: SkillRegistry,
    image: DecodedImage,
    flags,
    first: PrecheckResult,
    model: str,
) -> tuple[PrecheckResult, tuple[int, int]]:
    second, in_tokens, out_tokens = await llm.structured_call(
        system=build_recheck_system(skills, flags, first),
        image=image,
        user_text=PRECHECK_RECHECK_USER_TEXT,
        schema=PrecheckResult,
        max_tokens=PRECHECK_MAX_TOKENS,
        step="precheck",
        model=model,
    )
    return _merge_precheck(first, second), (in_tokens, out_tokens)


async def _reclassify(
    llm: Provider,
    skills: SkillRegistry,
    image: DecodedImage,
    flags,
    pre: PrecheckResult,
    model: str,
) -> tuple[PrecheckResult, tuple[int, int]]:
    """分类复核：只更新题材分类字段，AI 判定结果不动。"""
    result, in_tokens, out_tokens = await llm.structured_call(
        system=build_reclassify_system(skills, flags, pre),
        image=image,
        user_text=PRECHECK_RECLASSIFY_USER_TEXT,
        schema=PrecheckResult,
        max_tokens=PRECHECK_MAX_TOKENS,
        step="precheck",
        model=model,
    )
    updated = pre.model_copy()
    updated.category = result.category
    updated.category_confidence = result.category_confidence
    updated.category_reason = result.category_reason
    return updated, (in_tokens, out_tokens)


def _merge_precheck(first: PrecheckResult, second: PrecheckResult) -> PrecheckResult:
    """两轮结论一致保留原结论，不一致则降为 uncertain；置信度取均值。

    复核只针对 AI 判定：题材分类沿用第一轮（复核 prompt 聚焦 AI，
    第二轮的分类输出不可靠，不作为路由依据）。
    """
    merged = second.model_copy()
    merged.is_ai = first.is_ai if first.is_ai == second.is_ai else "uncertain"
    merged.ai_confidence = (first.ai_confidence + second.ai_confidence) / 2
    merged.category = first.category
    merged.category_confidence = first.category_confidence
    merged.category_reason = first.category_reason
    return merged


def _ai_reason(pre: PrecheckResult) -> str:
    hits = [f"{e.aspect}：{e.finding}" for e in pre.ai_evidence if e.supports_ai]
    if pre.is_ai == "ai":
        return "；".join(hits) or pre.description
    if pre.is_ai == "uncertain":
        return ("疑点：" + "；".join(hits)) if hits else "证据不足，无法确定"
    return "未发现典型 AI 生成特征"
