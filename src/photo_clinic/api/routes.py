"""HTTP 路由：POST /api/review、GET /health、GET /skills + 异常映射。"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from photo_clinic.config import Config
from photo_clinic.metadata import (
    ImageTooLargeError,
    InvalidImageError,
    InvalidMediaTypeError,
    c2pa_available,
)
from photo_clinic.pipeline import run_review
from photo_clinic.providers.base import Provider, UpstreamError
from photo_clinic.schemas import ReviewRequest, ReviewResponse
from photo_clinic.skills import SkillRegistry


class AccessDeniedError(Exception):
    """缺少或错误的访问密钥。"""


def _make_auth(access_key: str | None):
    """设置 server_access_key 后 /api/review 要求 X-API-Key 请求头；未设置 = 无鉴权。"""

    async def check(x_api_key: str | None = Header(default=None)) -> None:
        if access_key and x_api_key != access_key:
            raise AccessDeniedError("缺少或错误的访问密钥（X-API-Key）")

    return check


def create_router(config: Config, skills: SkillRegistry, llm: Provider) -> APIRouter:
    router = APIRouter()
    auth_check = _make_auth(config.server_access_key)
    # 并发闸门：LLM 调用耗时长，限制同时在途的评审请求数，防止大图请求堆叠撑爆内存
    semaphore = asyncio.Semaphore(config.max_concurrent_reviews)

    @router.post("/api/review", response_model=ReviewResponse)
    async def review(request: Request, _: None = Depends(auth_check)) -> ReviewResponse:
        async with semaphore:
            # 先拿闸门再读请求体：排队中的请求不持有已解析的 base64 大字符串
            try:
                request_body = ReviewRequest.model_validate(await request.json())
            except ValidationError as exc:
                raise RequestValidationError(exc.errors()) from exc
            except json.JSONDecodeError as exc:
                raise RequestValidationError(
                    [{"loc": ["body"], "msg": f"JSON 解析失败: {exc}", "type": "json_invalid"}]
                ) from exc
            return await run_review(request_body, config=config, skills=skills, llm=llm)

    @router.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model": config.model,
            "provider": config.provider,
            "skills": skills.names(),
            "c2pa_available": c2pa_available(),
        }

    @router.get("/skills")
    async def list_skills() -> list[dict]:
        return [
            {"name": name, "description": skills.get(name).description} for name in skills.names()
        ]

    return router


def _make_error_handler(status: int, code: str):
    async def handler(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": str(exc)}})

    return handler


def register_exception_handlers(app: FastAPI) -> None:
    """FastAPI 0.141 起异常处理只在 app 层注册（APIRouter 已无该 API）。"""
    app.add_exception_handler(InvalidImageError, _make_error_handler(400, "invalid_image"))
    app.add_exception_handler(InvalidMediaTypeError, _make_error_handler(400, "invalid_media_type"))
    app.add_exception_handler(ImageTooLargeError, _make_error_handler(413, "image_too_large"))
    app.add_exception_handler(UpstreamError, _make_error_handler(502, "upstream_error"))
    app.add_exception_handler(AccessDeniedError, _make_error_handler(401, "unauthorized"))
