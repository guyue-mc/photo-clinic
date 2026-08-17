"""FastAPI 应用工厂。

启动：uvicorn photo_clinic.server:create_app --factory --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI

from photo_clinic import __version__
from photo_clinic.config import Config, load_config
from photo_clinic.llm import build_llm
from photo_clinic.providers.base import Provider
from photo_clinic.registry import REQUIRED_SKILLS
from photo_clinic.skills import SkillRegistry, SkillsError


def create_app(config: Config | None = None, llm: Provider | None = None) -> FastAPI:
    config = config or load_config()
    skills = SkillRegistry(config.skills_dir)
    skills.load()  # 目录缺失/SKILL.md 损坏 → SkillsError，启动期 fail fast
    missing = [name for name in REQUIRED_SKILLS if name not in skills.names()]
    if missing:
        raise SkillsError(f"缺少必需 skill: {missing}（skills 目录：{config.skills_dir}）")
    llm = llm or build_llm(config)  # 缺 API Key 等配置错误也在此 fail fast

    from photo_clinic.api.routes import create_router, register_exception_handlers

    app = FastAPI(title="photo-agent", version=__version__)
    register_exception_handlers(app)
    if config.allowed_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allowed_origins,
            allow_methods=["POST", "GET"],
            allow_headers=["Content-Type", "X-API-Key"],
        )
    app.include_router(create_router(config, skills, llm))
    return app
