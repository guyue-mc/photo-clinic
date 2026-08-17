"""HTTP 层：路由、错误映射、/health、/skills。"""
from __future__ import annotations

import base64

import pytest
from conftest import PRECHECK_LANDSCAPE, REVIEW
from fastapi.testclient import TestClient

from photo_clinic.config import Config
from photo_clinic.providers.base import UpstreamError
from photo_clinic.server import create_app


def make_config(tmp_path) -> Config:
    return Config(
        provider="openai_compat",
        base_url="https://example.com/v1",
        api_key="k",
        model="test-model",
        max_image_mb=10,
        skills_dir=tmp_path / ".claude" / "skills",
    )


@pytest.fixture
def client(skills, tmp_path, fake_provider):
    app = create_app(config=make_config(tmp_path), llm=fake_provider)
    return TestClient(app)


def test_review_landscape_200(client, fake_provider, jpeg_image):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    resp = client.post("/api/review", json={"image_base64": jpeg_image})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "landscape"
    assert body["review"]["skill"] == "landscape-review"
    assert body["ai_detection"]["verdict"] == "not_ai"
    assert body["model"] == "test-model"


def test_review_media_type_mismatch_400(client, jpeg_image):
    resp = client.post(
        "/api/review", json={"image_base64": jpeg_image, "media_type": "image/png"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_media_type"


def test_review_non_image_400(client):
    payload = base64.b64encode(b"not an image at all").decode()
    resp = client.post("/api/review", json={"image_base64": payload})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_image"


def test_review_over_hard_limit_413(skills, tmp_path):
    config = make_config(tmp_path)
    small = Config(
        provider=config.provider,
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        max_image_mb=1,
        skills_dir=config.skills_dir,
    )
    app = create_app(config=small, llm=object())  # llm 不会被用到
    client = TestClient(app)
    # 硬上限 = max × 5 = 5MB；软上限（1MB）内的超大图会走自动压缩而非 413
    payload = base64.b64encode(b"A" * (5 * 1024 * 1024 + 100)).decode()
    resp = client.post("/api/review", json={"image_base64": payload})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "image_too_large"


def test_review_auto_compresses_oversized_image(skills, tmp_path, fake_provider, medium_noise_png):
    config = make_config(tmp_path)
    small = Config(
        provider=config.provider,
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        max_image_mb=1,  # 3.5MB 图 > 1MB 软上限 → 自动压缩后继续评审
        skills_dir=config.skills_dir,
    )
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    app = create_app(config=small, llm=fake_provider)
    client = TestClient(app)
    resp = client.post("/api/review", json={"image_base64": medium_noise_png})
    assert resp.status_code == 200
    assert resp.json()["route"] == "landscape"


def test_review_missing_field_422(client):
    resp = client.post("/api/review", json={})
    assert resp.status_code == 422


def test_upstream_error_502(skills, tmp_path, jpeg_image):
    class Exploding:
        async def structured_call(self, **kwargs):
            raise UpstreamError("api_error", "boom")

    app = create_app(config=make_config(tmp_path), llm=Exploding())
    client = TestClient(app)
    resp = client.post("/api/review", json={"image_base64": jpeg_image})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_error"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "test-model"
    assert "ai-detection" in body["skills"]
    assert "c2pa_available" in body


def make_client_with_access_key(skills, tmp_path, fake_provider, key: str) -> TestClient:
    config = make_config(tmp_path)
    secured = Config(
        provider=config.provider,
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        max_image_mb=config.max_image_mb,
        skills_dir=config.skills_dir,
        server_access_key=key,
    )
    app = create_app(config=secured, llm=fake_provider)
    return TestClient(app)


def test_review_requires_access_key_when_configured(skills, tmp_path, fake_provider, jpeg_image):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    client = make_client_with_access_key(skills, tmp_path, fake_provider, "secret-key")
    resp = client.post("/api/review", json={"image_base64": jpeg_image})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
    assert fake_provider.calls == []


def test_review_accepts_correct_access_key(skills, tmp_path, fake_provider, jpeg_image):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    client = make_client_with_access_key(skills, tmp_path, fake_provider, "secret-key")
    resp = client.post(
        "/api/review",
        json={"image_base64": jpeg_image},
        headers={"X-API-Key": "secret-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["route"] == "landscape"


def test_review_rejects_wrong_access_key(skills, tmp_path, fake_provider, jpeg_image):
    fake_provider.script("precheck", PRECHECK_LANDSCAPE).script("review", REVIEW)
    client = make_client_with_access_key(skills, tmp_path, fake_provider, "secret-key")
    resp = client.post(
        "/api/review", json={"image_base64": jpeg_image}, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401
    assert fake_provider.calls == []


def test_health_open_without_access_key(skills, tmp_path, fake_provider):
    client = make_client_with_access_key(skills, tmp_path, fake_provider, "secret-key")
    assert client.get("/health").status_code == 200


def test_skills_listing(client):
    resp = client.get("/skills")
    assert resp.status_code == 200
    names = [item["name"] for item in resp.json()]
    assert {"ai-detection", "subject-classify", "landscape-review"} <= set(names)


def test_skills_listing_includes_descriptions(client):
    body = client.get("/skills").json()
    item = next(i for i in body if i["name"] == "ai-detection")
    assert item["description"] == "test skill ai-detection"


def test_review_data_uri_prefix_400(client, jpeg_image):
    resp = client.post(
        "/api/review", json={"image_base64": f"data:image/jpeg;base64,{jpeg_image}"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_image"


def test_review_empty_image_400(client):
    resp = client.post("/api/review", json={"image_base64": ""})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_image"
