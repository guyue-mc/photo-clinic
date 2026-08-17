"""schemas.py：字段校验、枚举约束、分数 clamp 边界。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from photo_clinic.schemas import (
    BranchReviewResult,
    MetadataFlags,
    PrecheckResult,
    ReviewRequest,
    Usage,
)

VALID_PRECHECK = {
    "is_ai": "not_ai",
    "ai_confidence": 50,
    "category": "landscape",
    "category_confidence": 80,
    "category_reason": "景观",
    "description": "山",
}


def test_precheck_confidence_clamped():
    assert PrecheckResult(**{**VALID_PRECHECK, "ai_confidence": 150}).ai_confidence == 100.0
    assert PrecheckResult(**{**VALID_PRECHECK, "ai_confidence": -5}).ai_confidence == 0.0
    assert (
        PrecheckResult(**{**VALID_PRECHECK, "category_confidence": 999}).category_confidence
        == 100.0
    )


def test_total_score_clamped():
    assert BranchReviewResult(total_score=15).total_score == 10.0
    assert BranchReviewResult(total_score=-2).total_score == 0.0
    assert BranchReviewResult(total_score=None).total_score is None
    assert BranchReviewResult().total_score is None


def test_dimension_scores_clamped():
    result = BranchReviewResult(
        dimensions=[
            {"dimension": "构图", "score": 12.0, "comment": "a"},
            {"dimension": "光线", "score": -1.0, "comment": "b"},
        ]
    )
    assert [d.score for d in result.dimensions] == [10.0, 0.0]


def test_review_request_missing_field_raises():
    with pytest.raises(ValidationError):
        ReviewRequest()


def test_precheck_unknown_verdict_raises():
    with pytest.raises(ValidationError):
        PrecheckResult(**{**VALID_PRECHECK, "is_ai": "maybe"})


def test_precheck_unknown_category_raises():
    with pytest.raises(ValidationError):
        PrecheckResult(**{**VALID_PRECHECK, "category": "food"})


def test_metadata_flags_defaults():
    flags = MetadataFlags()
    assert flags.c2pa_available is False
    assert flags.c2pa_ai_manifest is None
    assert flags.suspicious_software is None
    assert flags.has_parameters_chunk is False
    assert flags.evidence_strength == "none"


def test_usage_defaults():
    usage = Usage()
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
