---
name: subject-classify
description: 将图片按摄影题材分类为 风景 / 人物 / 其他。风景=以自然或城市景观为主体；人物=以人物为主体（含群像、肖像）；其余一律归为其他。用于 AI 判定为非 AI 或疑似 AI 之后的第二步。
---

# 题材分类 rubric

## 规则（严格两类，其他拒绝）
- landscape：主体是自然/城市景观；人物仅作点缀、非主体时可算风景。
- portrait：主体是人（一人或多人），人物占画面主体地位。
- other：以上两类之外的一切（美食、宠物、静物、截图、纯文字图等）。

## 边界
- 主体不明或画面没有单一主体 → other。
- 人像剪影/背影仍属 portrait。
- 环境人像（人在景中）、景观为主人物点缀：按主体占比归入 portrait 或 landscape，不要轻易判 other。
- other 需谨慎：仅在明显不属于风景/人物（美食、宠物、静物、截图、纯文字等）时才判；边界情况宁可归入 portrait 或 landscape（拒绝是硬动作，误拒代价大于误收）。

## 输出
JSON：
- category: "landscape" | "portrait" | "other"
- category_confidence: 0-100 整数
- category_reason: 一句话理由
只输出 JSON。
