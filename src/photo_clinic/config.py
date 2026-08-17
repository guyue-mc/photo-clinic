"""配置：从环境变量（含 .env 文件）加载。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

PROVIDERS = ("openai_compat", "anthropic")

# 默认值按 provider 区分：openai_compat 默认通义百炼，anthropic 默认 Opus
_DEFAULT_MODEL = {"openai_compat": "qwen3-vl-32b-instruct", "anthropic": "claude-opus-5"}
_DEFAULT_BASE_URL = {"openai_compat": "https://dashscope.aliyuncs.com/compatible-mode/v1"}


@dataclass(frozen=True)
class Config:
    provider: str
    base_url: str | None
    api_key: str | None
    model: str
    max_image_mb: int
    skills_dir: Path
    # 设置后 /api/review 要求请求头 X-API-Key（内测/公开部署用；未设置 = 无鉴权，本地开发）
    server_access_key: str | None = None
    # CORS 允许的来源（逗号分隔）；空 = 不加 CORS 中间件
    allowed_origins: list[str] = field(default_factory=list)

    @property
    def max_image_bytes(self) -> int:
        return self.max_image_mb * 1024 * 1024


def _load_dotenv() -> None:
    """加载 .env（不覆盖已存在的环境变量）。只支持 KEY=VALUE 行。"""
    for path in (Path.cwd() / ".env", _REPO_ROOT / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config() -> Config:
    _load_dotenv()
    provider = os.environ.get("PHOTO_AGENT_PROVIDER", "openai_compat")
    if provider not in PROVIDERS:
        raise ValueError(f"未知 PHOTO_AGENT_PROVIDER: {provider}（可选：{', '.join(PROVIDERS)}）")
    skills_dir = Path(
        os.environ.get("PHOTO_AGENT_SKILLS_DIR") or (_REPO_ROOT / ".claude" / "skills")
    )
    allowed_origins = [
        origin.strip()
        for origin in os.environ.get("PHOTO_AGENT_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return Config(
        provider=provider,
        base_url=os.environ.get("PHOTO_AGENT_BASE_URL") or _DEFAULT_BASE_URL.get(provider),
        api_key=os.environ.get("PHOTO_AGENT_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
        model=os.environ.get("PHOTO_AGENT_MODEL") or _DEFAULT_MODEL[provider],
        max_image_mb=int(os.environ.get("PHOTO_AGENT_MAX_IMAGE_MB") or "10"),
        skills_dir=skills_dir,
        server_access_key=os.environ.get("PHOTO_AGENT_ACCESS_KEY") or None,
        allowed_origins=allowed_origins,
    )
