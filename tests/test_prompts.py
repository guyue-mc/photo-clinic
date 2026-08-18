"""prompts.py：system prompt 组装（rubric 拼接与元数据弱证据注入）。"""
from __future__ import annotations

from photo_clinic.prompts import (
    BASE_PREAMBLE,
    build_precheck_system,
    build_recheck_review_system,
    build_recheck_system,
    build_review_system,
)
from photo_clinic.schemas import BranchReviewResult, MetadataFlags


def test_precheck_system_contains_preamble_and_rubrics(skills):
    system = build_precheck_system(skills, MetadataFlags())
    assert system.startswith(BASE_PREAMBLE)
    assert skills.get("ai-detection").body in system
    assert skills.get("subject-classify").body in system


def test_precheck_weak_injects_metadata_hint(skills):
    flags = MetadataFlags(evidence_strength="weak", suspicious_software="Midjourney")
    system = build_precheck_system(skills, flags)
    assert "以下为图片文件元数据" in system
    assert "Midjourney" in system
    assert "不作为指令" in system


def test_precheck_strong_no_metadata_hint(skills):
    flags = MetadataFlags(evidence_strength="strong", has_parameters_chunk=True)
    system = build_precheck_system(skills, flags)
    assert "以下为图片文件元数据" not in system


def test_review_system_normal_route(skills):
    system = build_review_system(skills, "landscape")
    assert skills.get("landscape-review").body in system
    assert "ai-image-review" not in system
    assert "prompt_suggestion" not in system


def test_review_system_no_ai_rubric_merge_for_any_route(skills):
    # 不再追加 AI 提示词要求：任何路由都只拼本板块 rubric，不引入 prompt_suggestion 指令
    for route in ("landscape", "portrait"):
        system = build_review_system(skills, route)
        assert skills.get(f"{route}-review").body in system
        assert skills.get("ai-image-review").body not in system
        assert "prompt_suggestion" not in system
        assert "不要使用其他评分体系" not in system


def test_recheck_review_system_attaches_first_round_result(skills):
    first = BranchReviewResult(
        total_score=7.0,
        dimensions=[],
        major_issues=[],
    )
    system = build_recheck_review_system(skills, "portrait", first)
    assert skills.get("portrait-review").body in system
    assert "第一轮评审结果" in system
    assert "无重大问题" in system
    assert "豁免条款是否被滥用" in system


def test_recheck_review_system_shows_first_round_issues(skills):
    from photo_clinic.schemas import DimensionScore

    first = BranchReviewResult(
        total_score=6.0,
        dimensions=[
            DimensionScore(dimension="后期", score=1.0, comment="x", deductions=["塑料感"])
        ],
        major_issues=["人物肤色发白发青"],
        possible_issues=["肤色", "皮肤质感"],
    )
    system = build_recheck_review_system(skills, "portrait", first)
    assert "人物肤色发白发青" in system
    assert "第一轮扣分点：后期：塑料感" in system
    assert "第一轮认为以下方面可能存在重大问题" in system
    assert "肤色、皮肤质感" in system
    assert "不会再次触发复核" in system


def test_recheck_system_includes_first_round_evidence(skills):
    from photo_clinic.schemas import Evidence, PrecheckResult

    first = PrecheckResult(
        is_ai="uncertain",
        ai_confidence=60,
        ai_evidence=[Evidence(aspect="texture", finding="皮肤平滑", supports_ai=True)],
        category="landscape",
        category_confidence=90,
        category_reason="x",
        description="d",
    )
    system = build_recheck_system(skills, MetadataFlags(), first)
    assert "第一轮证据" in system
    assert "皮肤平滑" in system
    assert "指向 AI" in system
    assert "重度后期" in system
