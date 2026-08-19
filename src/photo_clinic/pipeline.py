"""评审编排：预检 → 路由 → 板块评审（含疑似 AI 路径）。"""
from __future__ import annotations

from photo_clinic.config import Config
from photo_clinic.metadata import (
    DecodedImage,
    InvalidMediaTypeError,
    decode_image,
    detect_pale_skin,
    downscale_image,
    inspect_metadata,
)
from photo_clinic.prompts import (
    PRECHECK_RECHECK_USER_TEXT,
    PRECHECK_RECLASSIFY_USER_TEXT,
    PRECHECK_USER_TEXT,
    REVIEW_RECHECK_USER_TEXT,
    REVIEW_USER_TEXT,
    build_precheck_system,
    build_recheck_review_system,
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
    # 送 LLM 的图统一压到 ≤3000px/≤2MB（元数据检测已在此前完成；重编码会丢弃 EXIF 等证据），
    # 保留皮肤纹理细节供模型判断，同时限制单请求内存放大倍数，多轮调用复用同一份小图
    image = downscale_image(image)
    if flags.evidence_strength == "strong":
        # 元数据实锤 → 短路判 AI，跳过 LLM 预检，直接 AI 板块评审
        ai_detection = AiDetection(
            verdict="ai",
            confidence=METADATA_AI_CONFIDENCE,
            reason="图片元数据包含 AI 生成证据（C2PA 签名 / PNG 参数块 / 生成软件字段）",
            metadata=flags,
            source="metadata",
        )
        review, tokens = await _review_branch(llm, skills, image, "ai", model=model)
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
        review, tokens = await _review_branch(llm, skills, image, "ai", model=model)
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

    review, tokens = await _review_branch(llm, skills, image, pre.category, model=model)
    _add_usage(usage, tokens)
    ai_suspicion = None
    # 仅置信度 ≥50% 才提示"疑似 AI"（低于此值视为普通照片，不出疑似点评）
    if pre.is_ai == "uncertain" and pre.ai_confidence >= 50.0:
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


# 重大问题 → 所属维度映射关键词（按 rubric 第 0 节词汇表）与封顶分
MAJOR_DIMENSION_KEYWORDS = {
    "构图": ("构图", "畸变", "裁切", "穿头", "倾斜", "平衡"),
    "光线": ("曝光", "过曝", "死黑", "光比", "阴影", "光线"),
    "后期": ("肤色", "塑料", "液化", "色调", "皮肤", "后期"),
}
MAJOR_DIMENSION_CAP = 1.0  # 重大问题命中后，所属维度得分封顶（狠狠扣分）
# 磨皮/皮肤质感类问题不进重大问题（按用户定案：最多扣分点），命中即从 major_issues 中剔除
TEXTURE_MAJOR_KEYWORDS = ("磨皮", "皮肤质感", "塑料感", "毛孔", "肌理")


def _strip_texture_major(result: BranchReviewResult) -> BranchReviewResult:
    """把磨皮/皮肤质感类条目从 major_issues 中剔除（含模型自行写入的）。"""
    kept = [
        issue
        for issue in result.major_issues
        if not any(k in issue for k in TEXTURE_MAJOR_KEYWORDS)
    ]
    if len(kept) == len(result.major_issues):
        return result
    return result.model_copy(update={"major_issues": kept})


SKIN_MAJOR_MESSAGE = "肤色异常：皮肤发白（偏白/惨白），缺乏血色，偏离真实人体肤色"
PIXEL_SKIN_MAJOR_MESSAGE = "肤色异常：皮肤发白，缺乏血色（像素检测确认）"


def _apply_skin_rule(result: BranchReviewResult) -> BranchReviewResult:
    """肤色规则判定：模型在 skin_report 中只报告事实，是否重大问题由规则决定。

    阈值：肤色发白（pale/white）且不满足豁免（exempt=False）即构成重大问题
    （cosplay/角色扮演不豁免）。命中且 major_issues 中尚无对应条目时补入标准文案。
    """
    report = result.skin_report
    if report is None:
        return result
    severe = report.whiteness in ("pale", "white") and not report.exempt
    if severe and not any("肤色" in issue or "发白" in issue for issue in result.major_issues):
        return result.model_copy(
            update={"major_issues": [*result.major_issues, SKIN_MAJOR_MESSAGE]}
        )
    return result


def _apply_major_score_cap(result: BranchReviewResult) -> BranchReviewResult:
    """重大问题 → 所属维度狠狠扣分：命中后该维度得分封顶 1.0，总分按扣后分值重算。"""
    if not result.major_issues:
        return result
    dims = {d.dimension: d for d in result.dimensions}
    capped = False
    for issue in result.major_issues:
        for dim_name, keywords in MAJOR_DIMENSION_KEYWORDS.items():
            if dim_name in dims and any(k in issue for k in keywords):
                dim = dims[dim_name]
                if dim.score > MAJOR_DIMENSION_CAP:
                    # 记录扣分前的原始小计与扣分额，供展示「常规项 X 分 − 重大扣 Y 分 = 实得 Z 分」
                    dim.original_score = dim.score
                    dim.major_deduction = dim.score - MAJOR_DIMENSION_CAP
                    dim.score = MAJOR_DIMENSION_CAP
                    capped = True
                break
    if capped and result.total_score is not None:
        result.total_score = sum(d.score for d in result.dimensions)
    return result


async def _review_branch(
    llm: Provider,
    skills: SkillRegistry,
    image: DecodedImage,
    route: str,
    *,
    model: str,
) -> tuple[BranchReview, tuple[int, int]]:
    result, in_tokens, out_tokens = await llm.structured_call(
        system=build_review_system(skills, route),
        image=image,
        user_text=REVIEW_USER_TEXT,
        schema=BranchReviewResult,
        max_tokens=REVIEW_MAX_TOKENS,
        step="review",
        model=model,
    )
    # 人物板块重大问题复核：首轮自评存在不确定方面（possible_issues 非空）、或任一维度
    # 有扣分点（说明画面存在瑕疵，值得复核）时触发一轮复核，一次性覆盖全部可能问题；
    # 复核为终轮，不再递归触发。单轮 LLM 视觉判断有波动（肤色/磨皮判定偶发漏判），
    # 复核轮判出重大问题时按更严格结论（复核轮）整体替换；复核轮无则沿用第一轮。
    # 完全无瑕疵且无不确定项的图跳过复核，省一次调用。其余板块无重大问题语义。
    if route == "portrait" and (
        result.possible_issues or any(d.deductions for d in result.dimensions)
    ):
        recheck, in2, out2 = await llm.structured_call(
            system=build_recheck_review_system(skills, route, result),
            image=image,
            user_text=REVIEW_RECHECK_USER_TEXT,
            schema=BranchReviewResult,
            max_tokens=REVIEW_MAX_TOKENS,
            step="review",
            model=model,
        )
        in_tokens += in2
        out_tokens += out2
        if recheck.major_issues:
            result = recheck
    result = _strip_texture_major(result)
    result = _apply_skin_rule(result)
    # 像素级肤色兜底：模型报告可能漂移（同图时而 natural 时而 pale），
    # 像素检测是确定性的——发白即判重大问题，不依赖模型感知
    if route == "portrait" and detect_pale_skin(image) and not any(
        "肤色" in issue or "发白" in issue for issue in result.major_issues
    ):
        result = result.model_copy(
            update={"major_issues": [*result.major_issues, PIXEL_SKIN_MAJOR_MESSAGE]}
        )
    result = _apply_major_score_cap(result)
    review = BranchReview(
        skill=ROUTE_TO_SKILL[route],
        total_score=result.total_score,
        dimensions=result.dimensions,
        improvements=result.improvements,
        bonus_notes=result.bonus_notes,
        prompt_suggestion=result.prompt_suggestion,
        major_issues=result.major_issues,
        texture_report=result.texture_report,
        skin_report=result.skin_report,
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
