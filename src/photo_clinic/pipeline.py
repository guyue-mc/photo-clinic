"""评审编排：预检 → 路由 → 板块评审（含疑似 AI 路径）。"""
from __future__ import annotations

from photo_clinic.config import Config
from photo_clinic.metadata import (
    DecodedImage,
    InvalidMediaTypeError,
    decode_image,
    detect_pale_skin,
    downscale_image,
    extract_focal_length,
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
    DimensionScore,
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
# 临界复核：置信度落在 [30,70] 区间，或判 ai 但置信不足 98% 时，
# 带第一轮证据再复核一次（抑制边界样本上单次判定的随机翻转；
# 高置信 AI 也可能因「推理型怀疑」误判，98% 以下一律复核）
RECHECK_CONF_LOW, RECHECK_CONF_HIGH = 30.0, 70.0
RECHECK_AI_CONF_MAX = 98.0
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
    # 实际焦距（EXIF）需在重编码前读取（downscale 会丢弃 EXIF）；无焦距则 None，走推断
    focal_length = extract_focal_length(image.data)
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
        review, tokens = await _review_branch(
            llm, skills, image, "ai", model=model, focal_length=focal_length
        )
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
        review, tokens = await _review_branch(
            llm, skills, image, "ai", model=model, focal_length=focal_length
        )
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

    review, tokens = await _review_branch(
        llm, skills, image, pre.category, model=model, focal_length=focal_length
    )
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
# 维度满分表（评分一致性校验用；ai 板块维度名不同，自动跳过）
DIMENSION_MAX = {"构图": 3.0, "光线": 4.0, "后期": 3.0}
# 每条扣分点自动降 0.5，最多降 1 分（防「写扣分点却给满分」绕过评分从严）
PER_DEDUCTION_PENALTY = 0.5
MAX_CONSISTENCY_PENALTY = 1.0
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


# 评语元素词表（L4 冲突锁：出彩声明与同维度扣分点/重大问题说同一元素 = 既夸又贬）
_REVIEW_ELEMENTS = (
    "前景", "背景", "手部", "人物", "主体", "视角", "机位", "光线", "阴影",
    "光比", "光影", "色彩", "色调", "皮肤", "头发", "发丝", "眼神光",
    "轮廓光", "构图", "平衡", "层次", "裁切",
)


def _conflicting_element(note: str, sources: list[str]) -> str | None:
    """出彩声明与同维度扣分点/重大问题是否说到同一元素，返回元素名或 None。"""
    for element in _REVIEW_ELEMENTS:
        if element in note and any(element in s for s in sources):
            return element
    return None


def _major_dimension(result: BranchReviewResult) -> str | None:
    """重大问题所属维度（按关键词映射；无匹配返回 None）。"""
    for issue in result.major_issues:
        for dim_name, keywords in MAJOR_DIMENSION_KEYWORDS.items():
            if any(k in issue for k in keywords):
                return dim_name
    return None


# 多层锁降分时写入的用户可读扣分点（不暴露内部机制，用评审语言说明降分原因）
_DEDUCTION_NO_EXCEPTIONAL = "无突出出彩之处，未达满分标准"


def _major_label(issue: str) -> str:
    """提取重大问题短标签（冒号/逗号前的片段，最长 12 字），用于扣分点指明具体问题。"""
    for sep in ("：", ":", "，", ","):
        if sep in issue:
            return issue.split(sep)[0][:12]
    return issue[:12]


def _append_deduction(dim: DimensionScore, text: str) -> None:
    if text not in dim.deductions:
        dim.deductions = [*dim.deductions, text]


def _enforce_score_consistency(result: BranchReviewResult) -> BranchReviewResult:
    """评分一致性校验（满分多层锁，全部由代码执行）：

    L1 无瑕疵：写了扣分点却给满分 → 按扣分点数量降分（每条 0.5，最多 1）；
    L2 出彩声明：无扣分点给满分但 exceptional_note 为空 → 降 0.5；
    L4 冲突锁：出彩声明与任一扣分点/重大问题共享元素（≥2 字）→ 既夸又贬 → 降 0.5；
    L3 背书锁：照片存在重大问题（major_issues 非空）时，其余维度满分须在
       full_mark_endorsements 中被复核轮显式背书，未背书 → 降 0.5；
    L5 复读锁：背书理由与出彩声明完全相同 → 视为复读糊弄，无效背书 → 降 0.5。

    每次降分同步写入用户可读的扣分点（分数与扣分点保持一致）。
    非满分维度不受影响；重大问题所属维度由封顶规则处理，不参与背书锁。
    """
    changed = False
    major_dim = _major_dimension(result) if result.major_issues else None
    for dim in result.dimensions:
        max_score = DIMENSION_MAX.get(dim.dimension)
        if max_score is None or dim.score < max_score:
            continue
        if dim.deductions:
            penalty = min(len(dim.deductions) * PER_DEDUCTION_PENALTY, MAX_CONSISTENCY_PENALTY)
            dim.score = max(max_score - penalty, 0.0)
            changed = True
            continue
        if not dim.exceptional_note:
            dim.score = max(max_score - PER_DEDUCTION_PENALTY, 0.0)
            _append_deduction(dim, _DEDUCTION_NO_EXCEPTIONAL)
            changed = True
            continue
        # L4 冲突锁：出彩声明与同维度扣分点/重大问题说同一元素 → 既夸又贬
        conflict_sources = list(dim.deductions)
        if dim.dimension == major_dim:
            conflict_sources.extend(result.major_issues)
        element = _conflicting_element(dim.exceptional_note, conflict_sources)
        if element:
            dim.score = max(max_score - PER_DEDUCTION_PENALTY, 0.0)
            dim.exceptional_note = None  # 撤销自相矛盾的出彩声明
            _append_deduction(dim, f"{element}：评语前后矛盾（既肯定又指出其问题）")
            changed = True
            continue
        if result.major_issues and dim.dimension != major_dim:
            endorsement = result.full_mark_endorsements.get(dim.dimension)
            if not endorsement or endorsement == dim.exceptional_note:
                # 扣分点先写该板块自身的问题（复核轮给出的具体瑕疵），再附评分从严说明
                critique = result.full_mark_critiques.get(dim.dimension) or "未达满分标准"
                dim.score = max(max_score - PER_DEDUCTION_PENALTY, 0.0)
                _append_deduction(
                    dim,
                    f"{critique}（照片存在重大问题（{_major_label(result.major_issues[0])}），其余维度评分从严）",
                )
                changed = True
    if changed and result.total_score is not None:
        result.total_score = sum(d.score for d in result.dimensions)
    return result


# 输出术语黑名单：rubric 内部术语不得出现在点评输出中（服务端强制剔除）
JARGON_TERMS = ("生命力四要素", "生命力要素", "动态的道具", "不确定的瞬间", "飞扬的发丝", "灵动的姿态")


def _strip_jargon(result: BranchReviewResult) -> BranchReviewResult:
    """从 bonus_notes 中剔除含内部术语的句子（术语禁令由代码强制执行）。"""
    if not result.bonus_notes:
        return result
    kept = [
        sentence.strip()
        for sentence in result.bonus_notes.replace("；", "。").replace(";", "。").split("。")
        if sentence.strip() and not any(term in sentence for term in JARGON_TERMS)
    ]
    cleaned = "。".join(kept)
    if cleaned != result.bonus_notes:
        return result.model_copy(update={"bonus_notes": cleaned or None})
    return result


# 焦段矛盾：建议中的焦段方向与画面透视特征冲突（压缩感=长焦特征、近大远小=广角特征）
_FOCAL_WIDE_WORDS = ("广角", "稍广角", "更广角", "广角镜头")
_FOCAL_TELE_WORDS = ("长焦", "更长焦", "长焦镜头")
_FOCAL_COMPRESSION = ("压缩感", "压缩背景", "扁平", "空间压缩")
_FOCAL_PERSPECTIVE = ("近大远小", "透视拉伸", "强烈透视", "透视夸张")


def _filter_focal_advice(
    result: BranchReviewResult, focal_mm: float | None = None
) -> BranchReviewResult:
    """剔除焦段方向错误的改进建议。

    实际焦距（EXIF）已知时：建议断言与实际焦距相反的焦段 → 剔除（如长焦图建议广角）。
    焦距未知时：剔除自相矛盾的建议（「有压缩感却建议广角」等内部矛盾）。
    """
    if focal_mm is not None:
        is_wide = focal_mm <= 35
        is_tele = focal_mm > 70
    else:
        is_wide = is_tele = None

    def _bad(item) -> bool:
        suggestion = item.suggestion
        if any(w in suggestion for w in _FOCAL_WIDE_WORDS):
            if is_wide is False:  # 实际非广角（标准或长焦）却建议广角
                return True
            if is_wide is None and any(c in suggestion for c in _FOCAL_COMPRESSION):
                return True
        if any(w in suggestion for w in _FOCAL_TELE_WORDS):
            if is_tele is False:  # 实际非长焦（标准或广角）却建议长焦
                return True
            if is_tele is None and any(p in suggestion for p in _FOCAL_PERSPECTIVE):
                return True
        return False

    changed = False
    for group in ("pre_shooting", "post_processing"):
        items = list(getattr(result.improvements, group))
        kept = [item for item in items if not _bad(item)]
        if len(kept) != len(items):
            result.improvements = result.improvements.model_copy(update={group: kept})
            changed = True
    return result if changed else result


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
    focal_length: float | None = None,
) -> tuple[BranchReview, tuple[int, int]]:
    result, in_tokens, out_tokens = await llm.structured_call(
        system=build_review_system(skills, route, focal_length),
        image=image,
        user_text=REVIEW_USER_TEXT,
        schema=BranchReviewResult,
        max_tokens=REVIEW_MAX_TOKENS,
        step="review",
        model=model,
    )
    # 人物板块重大问题复核：首轮自评存在不确定方面（possible_issues 非空）、任一维度
    # 有扣分点、或任一维度给满分（满分须两轮一致，防单轮虚高）时触发一轮复核，
    # 一次性覆盖全部可能问题；复核为终轮，不再递归触发。
    # 复核轮判出重大问题时按更严格结论（复核轮）整体替换；否则并入复核轮发现的额外
    # 扣分点（更严格），完全无瑕疵且无不确定项的图跳过复核，省一次调用。
    # 其余板块无重大问题语义。
    if route == "portrait" and (
        result.possible_issues
        or any(d.deductions for d in result.dimensions)
        or any(d.score >= DIMENSION_MAX.get(d.dimension, 0.0) for d in result.dimensions)
    ):
        recheck, in2, out2 = await llm.structured_call(
            system=build_recheck_review_system(skills, route, result, focal_length),
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
        else:
            # 并入复核轮发现的额外扣分点（按更严格结论合并）
            dims = {d.dimension: d for d in result.dimensions}
            changed = False
            for d in recheck.dimensions:
                if d.dimension in dims and d.deductions:
                    extra = [x for x in d.deductions if x not in dims[d.dimension].deductions]
                    if extra:
                        dims[d.dimension] = dims[d.dimension].model_copy(
                            update={"deductions": [*dims[d.dimension].deductions, *extra]}
                        )
                        changed = True
            if changed:
                result = result.model_copy(update={"dimensions": list(dims.values())})
    result = _strip_texture_major(result)
    result = _strip_jargon(result)
    result = _filter_focal_advice(result, focal_length)
    # 肤色重大问题以像素检测为唯一权威（双向门控）：模型自写/规则注入的肤色重大
    # 都需像素确认，未确认则剔除（防蓝光/冷调场景误报，如 5.jpg）；确认则确保有标准文案
    if route == "portrait":
        if detect_pale_skin(image):
            if not any("肤色" in issue or "发白" in issue for issue in result.major_issues):
                result = result.model_copy(
                    update={"major_issues": [*result.major_issues, SKIN_MAJOR_MESSAGE]}
                )
        else:
            kept = [
                issue
                for issue in result.major_issues
                if not ("肤色" in issue or "发白" in issue)
            ]
            if len(kept) != len(result.major_issues):
                result = result.model_copy(update={"major_issues": kept})
    result = _enforce_score_consistency(result)
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
