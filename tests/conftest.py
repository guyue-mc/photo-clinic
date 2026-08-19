"""pytest 公共设施：内存图片 fixtures + 临时 skills。"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from photo_clinic.skills import SkillRegistry

# 与 registry.py 一致的最小 skill 集合（测试用）
SKILL_NAMES = (
    "ai-detection",
    "subject-classify",
    "ai-image-review",
    "landscape-review",
    "portrait-review",
)


@pytest.fixture
def skills(tmp_path: Path) -> SkillRegistry:
    skills_dir = tmp_path / ".claude" / "skills"
    for name in SKILL_NAMES:
        d = skills_dir / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: test skill {name}\n---\n\n# {name} 正文占位\n",
            encoding="utf-8",
        )
    registry = SkillRegistry(skills_dir)
    registry.load()
    return registry


# 预检/评审的脚本化 LLM 响应（test_pipeline 与 test_api 共用）
PRECHECK_AI = {
    "is_ai": "ai",
    "ai_confidence": 99,  # 高于复核阈值 98，路由类测试只走一次预检
    "ai_evidence": [{"aspect": "hand_anatomy", "finding": "六根手指", "supports_ai": True}],
    "category": "other",
    "category_confidence": 0,
    "category_reason": "",
    "description": "一张人像",
}
PRECHECK_LANDSCAPE = {
    "is_ai": "not_ai",
    "ai_confidence": 5,
    "ai_evidence": [],
    "category": "landscape",
    "category_confidence": 92,
    "category_reason": "自然景观为主体",
    "description": "一张山景",
}
PRECHECK_PORTRAIT = {**PRECHECK_LANDSCAPE, "category": "portrait", "category_reason": "人物为主体"}
PRECHECK_OTHER = {**PRECHECK_LANDSCAPE, "category": "other", "category_reason": "美食照"}
PRECHECK_UNCERTAIN_LANDSCAPE = {
    **PRECHECK_LANDSCAPE,
    "is_ai": "uncertain",
    "ai_confidence": 72,  # 落在复核区间 [30,70] 之外，保证路由类测试只走一次预检
    "ai_evidence": [{"aspect": "texture", "finding": "天空纹理可疑", "supports_ai": True}],
}
REVIEW = {
    "total_score": 7.5,
    "dimensions": [
        {"dimension": "构图", "score": 2.5, "comment": "尚可", "deductions": ["地平线歪斜"]}
    ],
    "improvements": {
        "pre_shooting": [{"aspect": "机位", "suggestion": "低机位仰拍"}],
        "post_processing": [{"aspect": "曝光", "suggestion": "提亮阴影"}],
    },
    "prompt_suggestion": "改进后的提示词",
}


class FakeProvider:
    """按 step 分派脚本化响应；记录全部调用（system/step/schema/model）。

    script(step, payload) 接受 dict 或 list[dict]：list 时按调用顺序逐个弹出
    （用于复核类多次调用），最后一个保留复用；dict 等价于单元素列表。
    """

    def __init__(self) -> None:
        self.responses: dict[str, list[dict]] = {}
        self.calls: list[dict] = []

    def script(self, step: str, payload: dict | list[dict]) -> FakeProvider:
        self.responses[step] = list(payload) if isinstance(payload, list) else [payload]
        return self

    async def structured_call(self, *, system, image, user_text, schema, max_tokens, step, model):
        self.calls.append({"system": system, "step": step, "schema": schema, "model": model})
        payloads = self.responses[step]
        payload = payloads.pop(0) if len(payloads) > 1 else payloads[0]
        return schema.model_validate(payload), 100, 50


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture
def jpeg_image() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 60, 90)).save(buf, format="JPEG")
    return _b64(buf.getvalue())


def _random_rgb_png(width: int, height: int) -> str:
    """随机像素 PNG（体积 ≈ 宽×高×3 字节，确定可控），用于自动压缩路径测试。"""
    import os

    buf = io.BytesIO()
    Image.frombytes("RGB", (width, height), os.urandom(width * height * 3)).save(buf, format="PNG")
    return _b64(buf.getvalue())


@pytest.fixture
def large_noise_png() -> str:
    """3000x2000 随机 PNG ≈ 17MB（软上限之上、硬上限之下）。"""
    return _random_rgb_png(3000, 2000)


@pytest.fixture
def medium_noise_png() -> str:
    """1280x960 随机 PNG ≈ 3.5MB（软上限之上、硬上限之下）。"""
    return _random_rgb_png(1280, 960)


@pytest.fixture
def png_with_parameters() -> str:
    buf = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "Steps: 20, Sampler: DPM++ 2M, Model: sd_xl_base_1.0")
    Image.new("RGB", (64, 64), (30, 60, 90)).save(buf, format="PNG", pnginfo=info)
    return _b64(buf.getvalue())


@pytest.fixture
def jpeg_with_ai_software() -> str:
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), (30, 60, 90))
    exif = Image.Exif()
    exif[0x0131] = "Midjourney"  # IFD0 Software 标签
    img.save(buf, format="JPEG", exif=exif)
    return _b64(buf.getvalue())
