"""API 请求/响应与 LLM 输出契约（与 .claude/skills 中 SKILL.md 的输出字段一致）。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Route = Literal["ai", "landscape", "portrait", "rejected"]
SubjectCategory = Literal["landscape", "portrait", "other"]
AiVerdict = Literal["ai", "not_ai", "uncertain"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ReviewRequest(BaseModel):
    image_base64: str
    media_type: str | None = None
    model: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str


class Evidence(BaseModel):
    aspect: str
    finding: str
    supports_ai: bool


class MetadataFlags(BaseModel):
    c2pa_available: bool = False
    c2pa_ai_manifest: bool | None = None
    suspicious_software: str | None = None
    has_parameters_chunk: bool = False
    evidence_strength: Literal["none", "weak", "strong"] = "none"


class AiDetection(BaseModel):
    verdict: AiVerdict
    confidence: float  # 是 AI 的概率 0.0~1.0
    reason: str
    checks: list[Evidence] = Field(default_factory=list)
    metadata: MetadataFlags = Field(default_factory=MetadataFlags)
    source: Literal["metadata", "llm"] = "llm"


class AiSuspicion(BaseModel):
    """预检结果为 uncertain（疑似 AI）但进入风景/人物评审时的提醒。"""

    warning: str


class SubjectClassification(BaseModel):
    category: SubjectCategory
    confidence: float  # 0.0~1.0
    reason: str


class Improvement(BaseModel):
    aspect: str
    suggestion: str


class StageAdvice(BaseModel):
    pre_shooting: list[Improvement] = Field(default_factory=list)
    post_processing: list[Improvement] = Field(default_factory=list)


class DimensionScore(BaseModel):
    dimension: str
    score: float  # 0-10
    comment: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class PrecheckResult(BaseModel):
    """预检 LLM 输出（ai-detection + subject-classify 合并）。"""

    is_ai: AiVerdict
    ai_confidence: float = 50.0  # 0-100
    ai_evidence: list[Evidence] = Field(default_factory=list)
    category: SubjectCategory = "other"
    category_confidence: float = 50.0  # 0-100
    category_reason: str = ""
    description: str = ""

    @field_validator("ai_confidence", "category_confidence")
    @classmethod
    def _clamp_0_100(cls, value: float) -> float:
        return _clamp(value, 0.0, 100.0)


class BranchReviewResult(BaseModel):
    """板块评审 LLM 输出（三板块共用）。"""

    total_score: float | None = None  # 0-10；评分体系后定，先可空
    dimensions: list[DimensionScore] = Field(default_factory=list)
    improvements: StageAdvice = Field(default_factory=StageAdvice)
    bonus_notes: str | None = None  # 加分/前提项观察（不计入总分，人物板块使用）
    prompt_suggestion: str | None = None  # 仅 AI 板块使用

    @field_validator("total_score")
    @classmethod
    def _clamp_total(cls, value: float | None) -> float | None:
        return None if value is None else _clamp(value, 0.0, 10.0)

    @field_validator("dimensions")
    @classmethod
    def _clamp_dimension_scores(cls, dims: list[DimensionScore]) -> list[DimensionScore]:
        for dim in dims:
            dim.score = _clamp(dim.score, 0.0, 10.0)
        return dims


class BranchReview(BranchReviewResult):
    """板块评审结果（含路由到的 skill 名，由 pipeline 填充）。"""

    skill: str


class ReviewResponse(BaseModel):
    route: Route
    route_confidence: float  # 0.0~1.0
    ai_detection: AiDetection
    subject: SubjectClassification | None = None  # route=ai 时为 None
    review: BranchReview | None = None  # route=rejected 时为 None
    ai_suspicion: AiSuspicion | None = None  # 疑似 AI 但进入板块评审时非空
    model: str
    usage: Usage = Field(default_factory=Usage)
