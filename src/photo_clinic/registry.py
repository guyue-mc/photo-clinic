"""板块路由注册表。

扩展新板块（如「美食」）的完整触点：
1. 新建 .claude/skills/<name>/SKILL.md
2. subject-classify rubric 增加对应类别定义
3. schemas.py 的 SubjectCategory 加枚举值
4. 本文件 ROUTE_TO_SKILL 加一行
pipeline 零改动。
"""

PRECHECK_SKILLS: tuple[str, ...] = ("ai-detection", "subject-classify")

ROUTE_TO_SKILL: dict[str, str] = {
    "ai": "ai-image-review",
    "landscape": "landscape-review",
    "portrait": "portrait-review",
}

REQUIRED_SKILLS: tuple[str, ...] = PRECHECK_SKILLS + tuple(ROUTE_TO_SKILL.values())
