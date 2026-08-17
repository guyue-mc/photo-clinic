"""SKILL.md 加载：剥离 YAML frontmatter，正文作为 rubric 注入 system prompt。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class SkillsError(Exception):
    """skills 目录缺失或 SKILL.md 解析失败。"""


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    body: str  # frontmatter 剥除后的正文
    path: Path


def parse_skill_md(text: str, path: Path) -> SkillSpec:
    """解析 SKILL.md：frontmatter 只取 name/description，其余键忽略（与 Claude Code 兼容）。"""
    body = text.strip()
    frontmatter: dict = {}
    lines = body.splitlines()
    if lines and lines[0].strip() == "---":
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is None:
            raise SkillsError(f"{path}: frontmatter 缺少结束分隔符 ---")
        try:
            loaded = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError as exc:
            raise SkillsError(f"{path}: frontmatter 解析失败: {exc}") from exc
        if isinstance(loaded, dict):
            frontmatter = loaded
        body = "\n".join(lines[end + 1 :]).strip()
    name = frontmatter.get("name")
    if not name:
        raise SkillsError(f"{path}: frontmatter 缺少 name 字段")
    return SkillSpec(
        name=name,
        description=frontmatter.get("description", ""),
        body=body,
        path=path,
    )


class SkillRegistry:
    """按目录名索引的 skill 集合。"""

    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir
        self._skills: dict[str, SkillSpec] = {}

    def load(self) -> None:
        if not self._skills_dir.is_dir():
            raise SkillsError(f"skills 目录不存在: {self._skills_dir}")
        self._skills.clear()
        for sub in sorted(self._skills_dir.iterdir()):
            if not sub.is_dir():
                continue
            skill_md = sub / "SKILL.md"
            if skill_md.is_file():
                spec = parse_skill_md(skill_md.read_text(encoding="utf-8"), skill_md)
                self._skills[sub.name] = spec

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(f"未找到 skill: {name}") from None

    def names(self) -> list[str]:
        return sorted(self._skills)
