"""skills.py：frontmatter 剥离、解析错误、注册表行为。"""
from __future__ import annotations

from pathlib import Path

import pytest

from photo_clinic.skills import SkillRegistry, SkillsError, parse_skill_md


def test_parse_strips_frontmatter():
    spec = parse_skill_md(
        "---\nname: foo\ndescription: bar\n---\n\n# 正文\n内容",
        Path("foo/SKILL.md"),
    )
    assert spec.name == "foo"
    assert spec.description == "bar"
    assert spec.body == "# 正文\n内容"
    assert "---" not in spec.body


def test_parse_missing_name_raises():
    with pytest.raises(SkillsError):
        parse_skill_md("---\ndescription: bar\n---\nbody", Path("x/SKILL.md"))


def test_parse_missing_end_delimiter_raises():
    with pytest.raises(SkillsError):
        parse_skill_md("---\nname: foo\nbody", Path("x/SKILL.md"))


def test_parse_bad_yaml_raises():
    with pytest.raises(SkillsError):
        parse_skill_md("---\nname: [unclosed\n---\nbody", Path("x/SKILL.md"))


def test_parse_no_frontmatter_raises():
    with pytest.raises(SkillsError):
        parse_skill_md("plain body without frontmatter", Path("x/SKILL.md"))


def test_parse_unknown_frontmatter_keys_ignored():
    spec = parse_skill_md(
        "---\nname: foo\ndescription: bar\nmodel: something\n---\nbody",
        Path("x/SKILL.md"),
    )
    assert spec.name == "foo"
    assert spec.body == "body"


def test_parse_crlf_line_endings():
    spec = parse_skill_md(
        "---\r\nname: foo\r\ndescription: bar\r\n---\r\nbody",
        Path("x/SKILL.md"),
    )
    assert spec.body == "body"


def test_parse_empty_frontmatter_raises():
    with pytest.raises(SkillsError):
        parse_skill_md("---\n---\nbody", Path("x/SKILL.md"))


def test_parse_description_defaults_empty():
    spec = parse_skill_md("---\nname: foo\n---\nbody", Path("x/SKILL.md"))
    assert spec.description == ""


def test_registry_ignores_dirs_without_skill_md(tmp_path):
    skills_dir = tmp_path / "skills"
    (skills_dir / "empty-dir").mkdir(parents=True)
    (skills_dir / "random.txt").write_text("not a skill", encoding="utf-8")
    registry = SkillRegistry(skills_dir)
    registry.load()
    assert registry.names() == []


def test_registry_reload_idempotent(skills):
    names = skills.names()
    skills.load()
    assert skills.names() == names


def test_registry_loads_all_skills(skills):
    assert set(skills.names()) == {
        "ai-detection",
        "subject-classify",
        "ai-image-review",
        "landscape-review",
        "portrait-review",
    }
    assert skills.get("ai-detection").body


def test_registry_missing_dir_raises(tmp_path):
    registry = SkillRegistry(tmp_path / "nope")
    with pytest.raises(SkillsError):
        registry.load()


def test_registry_unknown_name_raises(skills):
    with pytest.raises(KeyError):
        skills.get("nope")
