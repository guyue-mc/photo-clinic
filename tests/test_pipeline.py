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
    SKIN_MAJOR_MESSAGE,
    _apply_major_score_cap,
    _enforce_score_consistency,
    _filter_focal_advice,
    _strip_jargon,
    _strip_texture_major,
    run_review,
)
from photo_clinic.schemas import BranchReviewResult, DimensionScore, ReviewRequest, TextureReport


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
REVIEW_CLEAN = {**REVIEW, "dimensions": [{"dimension": "构图", "score": 2.5, "comment": "尚可"}]}
REVIEW_WITH_MAJOR = {
    **REVIEW_CLEAN,
    "total_score": 6.0,
    "major_issues": ["曝光硬伤：窗外大面积过曝死白，主体细节丢失"],
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


async def test_portrait_full_score_triggers_recheck_and_merges_deductions(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 满分须两轮一致：首轮构图 3.0 满分，复核轮发现扣分点 → 并入并自动降分
    first = {
        **REVIEW_CLEAN,
        "dimensions": [
            {"dimension": "构图", "score": 3.0, "comment": "无可指摘", "deductions": []},
            {"dimension": "光线", "score": 3.0, "comment": "尚可", "deductions": []},
            {"dimension": "后期", "score": 2.0, "comment": "尚可", "deductions": []},
        ],
    }
    second = {
        **first,
        "dimensions": [
            {"dimension": "构图", "score": 3.0, "comment": "复核", "deductions": ["前景手部遮挡过重"]},
            {"dimension": "光线", "score": 3.0, "comment": "尚可", "deductions": []},
            {"dimension": "后期", "score": 2.0, "comment": "尚可", "deductions": []},
        ],
    }
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script("review", [first, second])
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    rev_calls = [c for c in fake_provider.calls if c["step"] == "review"]
    assert len(rev_calls) == 2
    comp = next(d for d in resp.review.dimensions if d.dimension == "构图")
    assert "前景手部遮挡过重" in comp.deductions
    assert comp.score == 2.5  # 一致性校验自动降分
    assert resp.review.total_score == 2.5 + 3.0 + 2.0


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
    assert SKIN_MAJOR_MESSAGE in resp.review.major_issues


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


async def test_portrait_skin_major_stripped_when_pixel_not_confirmed(
    skills, tmp_path, fake_provider
):
    # 模型自写肤色重大但像素未确认（蓝光/冷调场景）→ 剔除，像素为唯一权威
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 400), (205, 155, 115)).save(buf, format="JPEG")
    warm_image = base64.b64encode(buf.getvalue()).decode()
    model_wrote = {**REVIEW_CLEAN, "major_issues": ["肤色发白，缺乏血色，构成重大问题"]}
    fake_provider.script("precheck", PRECHECK_PORTRAIT).script("review", model_wrote)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=warm_image))
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


def test_strip_jargon_removes_internal_terms():
    result = make_result(
        bonus_notes=(
            "模特表现力强。具备『生命力四要素』中的动态与不确定瞬间。"
            "手部前伸与血迹增强了叙事与互动感。妆造精致。"
        )
    )
    updated = _strip_jargon(result)
    assert "生命力四要素" not in updated.bonus_notes
    assert "不确定的瞬间" not in updated.bonus_notes
    assert "手部前伸与血迹增强了叙事与互动感" in updated.bonus_notes
    assert "妆造精致" in updated.bonus_notes  # 正常词汇不受影响


def test_filter_focal_advice_drops_contradiction():
    # 焦距未知时：剔除内部矛盾建议（压缩感+建议广角；近大远小+建议长焦）
    from photo_clinic.schemas import Improvement, StageAdvice

    result = make_result()
    result.improvements = StageAdvice(
        pre_shooting=[
            Improvement(aspect="焦段", suggestion="当前焦段呈现较强压缩感，可尝试使用稍广角镜头增强纵深"),
            Improvement(aspect="站位", suggestion="让人物居中避免贴边"),
        ],
        post_processing=[
            Improvement(aspect="透视", suggestion="近大远小明显，可尝试长焦镜头压缩"),
        ],
    )
    updated = _filter_focal_advice(result)
    kept = [i.suggestion for i in updated.improvements.pre_shooting]
    assert kept == ["让人物居中避免贴边"]
    assert updated.improvements.post_processing == []


def test_filter_focal_advice_with_known_focal_length():
    # 实际焦距已知（长焦 135mm）：建议广角 → 剔除；建议长焦 → 保留
    from photo_clinic.schemas import Improvement, StageAdvice

    result = make_result()
    result.improvements = StageAdvice(
        pre_shooting=[
            Improvement(aspect="焦段", suggestion="建议改用广角镜头增强透视张力"),
            Improvement(aspect="焦段", suggestion="可继续使用长焦压缩背景，突出主体"),
        ],
        post_processing=[],
    )
    updated = _filter_focal_advice(result, focal_mm=135.0)
    kept = [i.suggestion for i in updated.improvements.pre_shooting]
    assert kept == ["可继续使用长焦压缩背景，突出主体"]


def test_strip_texture_major_keeps_unrelated_issues():
    result = make_result(major_issues=["曝光硬伤：窗外大面积过曝死白"])
    assert _strip_texture_major(result) is result




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


def test_score_consistency_lowers_full_marks_with_deductions():
    # 写了扣分点却给满分 → 代码自动降分（每条 0.5，最多 1）
    result = scored_result(
        10.0,
        DimensionScore(dimension="构图", score=3.0, comment="x", deductions=["前景杂乱"]),
        DimensionScore(dimension="光线", score=4.0, comment="x", deductions=["边缘略硬", "光比偏大"]),
        DimensionScore(dimension="后期", score=3.0, comment="x", exceptional_note="影调层次出色"),
    )
    updated = _enforce_score_consistency(result)
    dims = {d.dimension: d.score for d in updated.dimensions}
    assert dims["构图"] == 2.5  # 1 条扣分点 → -0.5
    assert dims["光线"] == 3.0  # 2 条 → -1（封顶降幅）
    assert dims["后期"] == 3.0  # 有出彩声明 → 不动
    assert updated.total_score == 2.5 + 3.0 + 3.0


def test_score_consistency_full_marks_need_exceptional_note():
    # 无瑕疵给满分但缺出彩声明 → 降 0.5（满分 = 无瑕疵 + 出彩）
    result = scored_result(
        10.0,
        DimensionScore(dimension="构图", score=3.0, comment="x"),
        DimensionScore(dimension="光线", score=4.0, comment="x"),
        DimensionScore(dimension="后期", score=3.0, comment="x"),
    )
    updated = _enforce_score_consistency(result)
    dims = {d.dimension: d.score for d in updated.dimensions}
    assert dims == {"构图": 2.5, "光线": 3.5, "后期": 2.5}
    assert updated.total_score == 2.5 + 3.5 + 2.5


def test_score_consistency_endorsement_lock_for_major_photos():
    # 重大问题照片：其余维度满分须复核轮显式背书，未背书 → 降 0.5，
    # 扣分点写明具体瑕疵 + 重大问题标签
    result = scored_result(
        10.0,
        DimensionScore(dimension="构图", score=3.0, comment="x", exceptional_note="仰拍出彩"),
        DimensionScore(dimension="光线", score=4.0, comment="x", exceptional_note="光效出色"),
        DimensionScore(dimension="后期", score=3.0, comment="x", exceptional_note="影调统一"),
    )
    result.major_issues = ["肤色惨白，缺乏血色，构成重大问题"]
    result.full_mark_critiques = {"构图": "前景手部遮挡过重，主体表现受干扰"}
    updated = _enforce_score_consistency(result)
    dims = {d.dimension: d.score for d in updated.dimensions}
    assert dims["构图"] == 2.5  # 未背书 → -0.5
    assert dims["光线"] == 3.5
    assert dims["后期"] == 3.0  # 重大问题所属维度走封顶规则，不受背书锁影响
    comp = next(d for d in updated.dimensions if d.dimension == "构图")
    assert any(
        "前景手部遮挡过重" in d and "肤色惨白" in d for d in comp.deductions
    )  # 先写板块自身问题，再附重大问题标签


def test_score_consistency_conflict_lock_same_dimension():
    # 冲突锁（同维度）：出彩声明与同维度重大问题说同一元素（主体）→ 降 0.5，
    # 扣分点写具体元素，且自相矛盾的出彩声明被撤销
    result = scored_result(
        10.0,
        DimensionScore(dimension="构图", score=3.0, comment="x", exceptional_note="主体居中出彩"),
        DimensionScore(dimension="光线", score=4.0, comment="x", exceptional_note="光效出色"),
        DimensionScore(dimension="后期", score=3.0, comment="x", exceptional_note="影调统一"),
    )
    result.major_issues = ["构图硬伤：主体被切断，画面不完整"]
    updated = _enforce_score_consistency(result)
    comp = next(d for d in updated.dimensions if d.dimension == "构图")
    assert comp.score == 2.5
    assert any("主体" in d for d in comp.deductions)
    assert comp.exceptional_note is None


def test_score_consistency_conflict_lock_cross_dimension_not_conflict():
    # 跨维度说同一元素不算矛盾（构图夸手部引导、后期批手部颜色，属合理评审）
    result = scored_result(
        9.0,
        DimensionScore(dimension="构图", score=3.0, comment="x", exceptional_note="前景手部引导出彩"),
        DimensionScore(dimension="光线", score=3.5, comment="x", deductions=["阴影生硬"]),
        DimensionScore(
            dimension="后期", score=2.5, comment="x", deductions=["前景手部红色斑点突兀"]
        ),
    )
    updated = _enforce_score_consistency(result)
    comp = next(d for d in updated.dimensions if d.dimension == "构图")
    assert comp.score == 3.0  # 跨维度不触发冲突锁


def test_score_consistency_endorsement_lock_passes_when_endorsed():
    # 复核轮显式背书 → 满分保留
    result = scored_result(
        10.0,
        DimensionScore(dimension="构图", score=3.0, comment="x", exceptional_note="仰拍出彩"),
        DimensionScore(dimension="光线", score=4.0, comment="x", exceptional_note="光效出色"),
        DimensionScore(dimension="后期", score=3.0, comment="x", exceptional_note="影调统一"),
    )
    result.major_issues = ["肤色惨白，缺乏血色，构成重大问题"]
    result.full_mark_endorsements = {
        "构图": "低机位仰拍+前景引导线，复核确认无瑕疵且出彩",
        "光线": "侧逆光+斑驳光影，复核确认无瑕疵且出彩",
    }
    updated = _enforce_score_consistency(result)
    dims = {d.dimension: d.score for d in updated.dimensions}
    assert dims["构图"] == 3.0
    assert dims["光线"] == 4.0


def test_score_consistency_copycat_endorsement_invalid():
    # 复读锁：背书理由与出彩声明完全相同 → 无效背书 → 降 0.5
    result = scored_result(
        10.0,
        DimensionScore(dimension="构图", score=3.0, comment="x", exceptional_note="仰拍出彩"),
        DimensionScore(dimension="光线", score=4.0, comment="x", exceptional_note="光效出色"),
        DimensionScore(dimension="后期", score=3.0, comment="x", exceptional_note="影调统一"),
    )
    result.major_issues = ["肤色惨白，缺乏血色，构成重大问题"]
    result.full_mark_endorsements = {"构图": "仰拍出彩", "光线": "侧逆光复核确认"}  # 构图是复读
    updated = _enforce_score_consistency(result)
    dims = {d.dimension: d.score for d in updated.dimensions}
    assert dims["构图"] == 2.5  # 复读背书无效 → -0.5
    assert dims["光线"] == 4.0  # 真实背书 → 保留


def test_score_consistency_skips_non_full_marks():
    # 非满分维度不受影响
    result = scored_result(
        8.0,
        DimensionScore(dimension="构图", score=2.5, comment="x", deductions=["前景杂乱"]),
        DimensionScore(dimension="光线", score=3.5, comment="x"),
        DimensionScore(dimension="后期", score=2.0, comment="x"),
    )
    updated = _enforce_score_consistency(result)
    assert updated.total_score == 8.0


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


async def test_high_confidence_ai_still_rechecks_below_98(
    skills, jpeg_image, tmp_path, fake_provider
):
    # 95% 的 ai 判定也触发复核（推理型怀疑可能高置信误判，如 6.jpg 案例）
    first = {**PRECHECK_AI, "ai_confidence": 95}
    second = {**PRECHECK_LANDSCAPE, "ai_confidence": 5}
    fake_provider.script("precheck", [first, second]).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    pre_calls = [c for c in fake_provider.calls if c["step"] == "precheck"]
    assert len(pre_calls) == 3  # 预检 + AI 复核 + 分类复核（首轮 other 低置信）
    # 两轮分歧 → uncertain（均值 50），按题材路由到 landscape
    assert resp.ai_detection.verdict == "uncertain"
    assert resp.route == "landscape"


async def test_not_ai_skips_recheck(skills, jpeg_image, tmp_path, fake_provider):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    resp = await run(skills, tmp_path, fake_provider, ReviewRequest(image_base64=jpeg_image))
    assert len([c for c in fake_provider.calls if c["step"] == "precheck"]) == 1
    assert resp.route == "landscape"
