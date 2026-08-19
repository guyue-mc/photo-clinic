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


class TextureReport(BaseModel):
    """皮肤质感结构化报告（模型只如实报告事实，是否重大问题由服务端规则判定）。"""

    pore_visibility: Literal["clear", "partial", "none"] = "partial"  # 毛孔/肌理可见性
    texture_coverage: Literal["most", "half", "partial", "little"] = "half"  # 可见纹理覆盖范围（most 大部分 / half 约一半 / partial 少部分 / little 几乎无）
    plastic_level: Literal["none", "slight", "moderate", "obvious", "severe"] = "slight"  # 塑料/蜡质感程度（none 自然 / slight 轻微 / moderate 中等 / obvious 明显 / severe 严重）


class SkinReport(BaseModel):
    """肤色结构化报告（模型只如实报告事实，是否重大问题由服务端规则判定）。"""

    whiteness: Literal["natural", "pale", "white"] = "natural"  # 肤色发白程度：natural 自然有血色 / pale 偏白发白 / white 惨白无血色
    exempt: bool = False  # 豁免：皮肤本身为非人类色（蓝/紫/绿等）或整体强风格化（黑白/单色/强滤镜）；cosplay/角色扮演不豁免


class DimensionScore(BaseModel):
    dimension: str
    score: float  # 0-10（重大问题扣分后的实得分）
    comment: str
    deductions: list[str] = Field(default_factory=list)  # 扣分点列表（空 = 无扣分点）
    original_score: float | None = None  # 重大问题扣分前的原始小计（无重大扣分 = None）
    major_deduction: float | None = None  # 重大问题扣分额（无重大扣分 = None）
    exceptional_note: str | None = None  # 满分维度的「出彩声明」（无瑕疵≠满分；无法给出出彩之处时不得给满分）


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
    major_issues: list[str] = Field(default_factory=list)  # 重大问题列表（空 = 无）
    possible_issues: list[str] = Field(default_factory=list)  # 首轮自评可能存在问题的方面（非空才触发一轮复核；空 = 无需复核）
    texture_report: TextureReport | None = None  # 皮肤质感结构化报告（仅人物板块）
    skin_report: SkinReport | None = None  # 肤色结构化报告（仅人物板块）
    full_mark_endorsements: dict[str, str] = Field(default_factory=dict)  # 满分维度显式背书：维度名 → 背书理由（仅复核轮填写）
    full_mark_critiques: dict[str, str] = Field(default_factory=dict)  # 复核轮对未背书满分维度的具体瑕疵说明：维度名 → 该维度存在的问题（仅复核轮填写）

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
    texture_report: TextureReport | None = None  # 皮肤质感结构化报告（内部机制，不参与点评输出）
    skin_report: SkinReport | None = None  # 肤色结构化报告（内部机制，不参与点评输出）


class ReviewResponse(BaseModel):
    route: Route
    route_confidence: float  # 0.0~1.0
    ai_detection: AiDetection
    subject: SubjectClassification | None = None  # route=ai 时为 None
    review: BranchReview | None = None  # route=rejected 时为 None
    ai_suspicion: AiSuspicion | None = None  # 疑似 AI 但进入板块评审时非空
    model: str
    usage: Usage = Field(default_factory=Usage)
