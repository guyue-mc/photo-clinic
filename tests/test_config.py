"""config.py：默认值、env 覆盖优先级、.env 解析与空串边界。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import photo_clinic.config as config_mod
from photo_clinic.config import _REPO_ROOT, Config, load_config

ENV_VARS = (
    "PHOTO_AGENT_PROVIDER",
    "PHOTO_AGENT_BASE_URL",
    "PHOTO_AGENT_API_KEY",
    "ANTHROPIC_API_KEY",
    "PHOTO_AGENT_MODEL",
    "PHOTO_AGENT_MAX_IMAGE_MB",
    "PHOTO_AGENT_SKILLS_DIR",
    "PHOTO_AGENT_ACCESS_KEY",
    "PHOTO_AGENT_ALLOWED_ORIGINS",
)


@pytest.fixture
def isolated_env(monkeypatch):
    """隔离真实 .env：禁用 dotenv 加载并清空相关环境变量。"""
    monkeypatch.setattr(config_mod, "_load_dotenv", lambda: None)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults(isolated_env):
    cfg = load_config()
    assert cfg.provider == "openai_compat"
    assert cfg.model == "qwen3-vl-32b-instruct"
    assert cfg.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert cfg.api_key is None
    assert cfg.max_image_mb == 10
    assert cfg.skills_dir == _REPO_ROOT / ".claude" / "skills"


def test_env_overrides(isolated_env):
    skills = Path("somewhere") / "skills"
    isolated_env.setenv("PHOTO_AGENT_BASE_URL", "https://api.example.com/v1")
    isolated_env.setenv("PHOTO_AGENT_API_KEY", "sk-test")
    isolated_env.setenv("PHOTO_AGENT_MODEL", "custom-model")
    isolated_env.setenv("PHOTO_AGENT_MAX_IMAGE_MB", "25")
    isolated_env.setenv("PHOTO_AGENT_SKILLS_DIR", str(skills))
    cfg = load_config()
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.api_key == "sk-test"
    assert cfg.model == "custom-model"
    assert cfg.max_image_mb == 25
    assert cfg.skills_dir == skills


def test_unknown_provider_raises(isolated_env):
    isolated_env.setenv("PHOTO_AGENT_PROVIDER", "ollama")
    with pytest.raises(ValueError, match="未知 PHOTO_AGENT_PROVIDER"):
        load_config()


def test_anthropic_defaults(isolated_env):
    isolated_env.setenv("PHOTO_AGENT_PROVIDER", "anthropic")
    cfg = load_config()
    assert cfg.model == "claude-opus-5"
    assert cfg.base_url is None


def test_api_key_fallback_and_priority(isolated_env):
    isolated_env.setenv("ANTHROPIC_API_KEY", "ant-key")
    assert load_config().api_key == "ant-key"
    isolated_env.setenv("PHOTO_AGENT_API_KEY", "photo-key")
    assert load_config().api_key == "photo-key"


def test_max_image_bytes_property():
    cfg = Config(
        provider="openai_compat",
        base_url=None,
        api_key=None,
        model="m",
        max_image_mb=3,
        skills_dir=Path("."),
    )
    assert cfg.max_image_bytes == 3 * 1024 * 1024


def test_empty_model_env_falls_back_to_default(isolated_env):
    isolated_env.setenv("PHOTO_AGENT_MODEL", "")
    assert load_config().model == "qwen3-vl-32b-instruct"


def test_empty_max_image_mb_env_falls_back_to_default(isolated_env):
    isolated_env.setenv("PHOTO_AGENT_MAX_IMAGE_MB", "")
    assert load_config().max_image_mb == 10


def test_dotenv_parses_key_value_and_strips_quotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "TEST_CFG_KEY1=value1\n"
        "# comment line\n"
        "TEST_CFG_KEY2=\"quoted value\"\n"
        "TEST_CFG_KEY3='single quoted'\n"
        "\n"
        "no-equals-line\n",
        encoding="utf-8",
    )
    for name in ("TEST_CFG_KEY1", "TEST_CFG_KEY2", "TEST_CFG_KEY3"):
        monkeypatch.delenv(name, raising=False)
    config_mod._load_dotenv()
    assert os.environ["TEST_CFG_KEY1"] == "value1"
    assert os.environ["TEST_CFG_KEY2"] == "quoted value"
    assert os.environ["TEST_CFG_KEY3"] == "single quoted"


def test_dotenv_does_not_override_existing_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_CFG_KEY1", "existing")
    (tmp_path / ".env").write_text("TEST_CFG_KEY1=new\n", encoding="utf-8")
    config_mod._load_dotenv()
    assert os.environ["TEST_CFG_KEY1"] == "existing"


def test_load_config_reads_cwd_dotenv(tmp_path, monkeypatch):
    """cwd/.env 先于 repo/.env 加载，setdefault 先到先得。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("PHOTO_AGENT_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.delenv("PHOTO_AGENT_MODEL", raising=False)
    assert load_config().model == "dotenv-model"


def test_access_key_and_origins_defaults(isolated_env):
    cfg = load_config()
    assert cfg.server_access_key is None
    assert cfg.allowed_origins == []


def test_access_key_and_origins_env(isolated_env):
    isolated_env.setenv("PHOTO_AGENT_ACCESS_KEY", "secret-key")
    isolated_env.setenv("PHOTO_AGENT_ALLOWED_ORIGINS", "https://a.com, https://b.com")
    cfg = load_config()
    assert cfg.server_access_key == "secret-key"
    assert cfg.allowed_origins == ["https://a.com", "https://b.com"]


def test_origins_empty_parts_filtered(isolated_env):
    isolated_env.setenv("PHOTO_AGENT_ALLOWED_ORIGINS", ", ,")
    assert load_config().allowed_origins == []
