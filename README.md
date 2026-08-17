# PhotoClinic

摄影评审 Agent：图片进来 → 判定是否 AI 生成 →（AI）AI 板块评审 + 提示词改进建议；（非 AI）严格分类风景/人物（其他拒绝）→ 按板块 rubric 评分 + 给出改进方案（前期拍摄 / 后期修图分组）。

各板块的评判标准存放在 `.claude/skills/` 下的 SKILL.md 文件中（可版本化、单一事实来源），两个环境共用：

- **Claude Code 交互环境**：贴图后运行 `/photo-review`，走原生 skill 机制
- **API 服务器**：Python 代码读同一份 SKILL.md 注入 system prompt

## 架构

```
POST /api/review (base64 JSON)
 → 解码 + Pillow 校验（400/413）
 → 元数据检测（C2PA / EXIF / PNG tEXt）
   ├─ 强证据命中 → 短路判 AI（跳过 LLM 预检）
   └─ 未命中 → 【LLM 预检】AI 判定 + 题材分类（结构化输出）
        ├─ AI        → ai-image-review 板块评审
        ├─ 风景/人物 → 对应板块评审
        └─ 其他      → 拒绝（route=rejected）
```

LLM 层为 provider 适配器：默认 OpenAI 兼容接口（通义 Qwen3-VL / DeepSeek / 智谱 GLM / 豆包均可），`PHOTO_AGENT_PROVIDER=anthropic` 可切 Claude。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[c2pa]"        # Windows；c2pa 可选，装上才有 C2PA 检测
cp .env.example .env                            # 填入你的 API Key
.venv/Scripts/uvicorn photo_clinic.server:create_app --factory --port 8000
```

评审一张图（小图可直接 curl；Windows 下大图 base64 会超命令行长度上限，建议用 Python 或文件方式传参）：

```bash
curl -s http://127.0.0.1:8000/api/review -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(base64 -w0 照片.jpg)\"}"

# Python 客户端示例
# import base64, httpx
# b64 = base64.b64encode(open("照片.jpg", "rb").read()).decode()
# r = httpx.post("http://127.0.0.1:8000/api/review", json={"image_base64": b64}, timeout=120)
# print(r.json())
```

Claude Code 交互使用：把图片拖进对话框，运行 `/photo-review`。

## API

- `POST /api/review` — 请求 `{"image_base64": str, "media_type": str|null, "model": str|null}`，响应含 `route`（ai/landscape/portrait/rejected）、`ai_detection`（判定 + 置信度 + 元数据证据）、`review`（维度评分 + 前期/后期改进建议）。错误信封 `{"error": {code, message}}`。
- `GET /health` — 状态、当前模型、已加载 skills、c2pa 可用性
- `GET /skills` — skill 列表

完整 schema 见 `src/photo_clinic/schemas.py`。

## 输入输出示例

请求：

```json
{"image_base64": "<图片的 base64 编码>"}
```

响应（人物照片，节选）：

```json
{
  "route": "portrait",
  "ai_detection": {"verdict": "not_ai", "confidence": 0.9, "source": "llm"},
  "subject": {"category": "portrait", "confidence": 0.95},
  "review": {
    "skill": "portrait-review",
    "total_score": 8.0,
    "dimensions": [
      {"dimension": "构图", "score": 2.5, "comment": "…"},
      {"dimension": "光线", "score": 3.0, "comment": "…"},
      {"dimension": "后期", "score": 2.5, "comment": "…"}
    ],
    "improvements": {
      "pre_shooting": [{"aspect": "光线", "suggestion": "…"}],
      "post_processing": [{"aspect": "色彩", "suggestion": "…"}]
    }
  },
  "model": "qwen3-vl-32b-instruct",
  "usage": {"input_tokens": 13450, "output_tokens": 1652}
}
```

其他路由：AI 图 `route="ai"`（review 额外含 `prompt_suggestion` 提示词改进建议）；非风景/人物题材（如美食）`route="rejected"`（`review` 为 null）；疑似 AI 附 `ai_suspicion.warning` 提醒。

## 说明与声明

- **数据出境**：图片会随请求发往所配置的 LLM 服务商 API（评审的本质要求）。部署方使用自己的 API key 与账号，图片数据由服务商隐私政策约束。
- **鉴权（可选）**：设置 `PHOTO_AGENT_ACCESS_KEY` 后，`/api/review` 要求请求头 `X-API-Key`，校验失败返回 401；`/health`、`/skills` 保持开放。未设置 = 无鉴权（仅限本地开发）。内测/公开部署务必设置。
- **CORS（可选）**：设置 `PHOTO_AGENT_ALLOWED_ORIGINS`（逗号分隔）后允许对应来源的浏览器跨域调用。
- **自动压缩**：图片超过 `PHOTO_AGENT_MAX_IMAGE_MB`（默认 10MB）时自动压缩到长边 2000px 内再评审；超过 5 倍上限（默认 50MB）返回 413。压缩在内存中完成，原图不会被修改。
- **AI 检测边界**：元数据（C2PA/EXIF）命中是高置信信号；LLM 视觉判定输出的是置信度三档（ai / not_ai / uncertain），不是绝对结论。
- **评分体系**：人物/风景板块按 构图(3分)/光线(4分)/后期(3分) 评分（总分 10 分）；AI 图板块按 主体主题(3分)/特效场景(4分)/光效对比(3分) 评分（总分 10 分），另输出提示词改进建议。
