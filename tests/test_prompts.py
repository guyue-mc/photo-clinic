"""prompts.py：system prompt 组装（rubric 拼接与元数据弱证据注入）。"""
from __future__ import annotations

from photo_clinic.prompts import (
    BASE_PREAMBLE,
    build_precheck_system,
    build_recheck_system,
    build_review_system,
)
from photo_clinic.schemas import MetadataFlags


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
    system = build_review_system(skills, "landscape", suspect_ai=False)
    assert skills.get("landscape-review").body in system
    assert "ai-image-review" not in system
    assert "prompt_suggestion" not in system


def test_review_system_suspect_ai_adds_prompt_requirement(skills):
    system = build_review_system(skills, "portrait", suspect_ai=True)
    assert skills.get("portrait-review").body in system
    # 不拼接 ai-image-review 完整 rubric（避免两套评分体系冲突），只追加针对性要求
    assert skills.get("ai-image-review").body not in system
    assert "prompt_suggestion" in system
    assert "不要使用其他评分体系" in system


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
