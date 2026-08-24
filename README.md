# PhotoClinic

PhotoClinic 是一个摄影评审 Agent：将摄影师的经验与审美标准蒸馏为严密的评审体系，对每张照片完成 AI 判定、题材分类与板块化评分，精准指出问题与改进空间，并给出前期拍摄与后期修图的分组建议。
流程：图片进来 → 判定是否 AI 生成 →（AI）AI 板块评审 + 提示词改进建议；（非 AI 或疑似 AI）严格分类风景/人物（其他拒绝）→ 按板块 rubric 评分 + 给出改进方案（前期拍摄 / 后期修图分组）；疑似 AI 附「如果是AI生图」提示词改进建议。

## 评审特性（v0.1.4）

- **重大问题排查**：评分前先排查重大问题（肤色发白、曝光硬伤、构图硬伤等），命中后首先点出，并在所属维度狠狠扣分（封顶 1.0），展示「常规 X-Y（重大问题扣分）= Z」
- **满分多层锁**：满分须通过无瑕疵、出彩声明、两轮一致、重大问题背书等五道代码锁，模型无法绕过
- **双检复核**：首轮存在扣分点/不确定项/满分维度时，触发一轮独立复核，按更严格结论合并
- **像素级肤色判定**：皮肤发白由像素检测确定（发白占比 >15% 且正常肤色占比 <5%），不依赖模型感知；蓝色主导场景（水体/天空）自动跳过防误报
- **焦段治理**：EXIF 焦距优先读取注入提示词；广角使用得当（无畸变/有张力）时不再推荐换人眼焦段
- **术语过滤**：评语中的内部术语（生命力四要素等）由服务端强制剔除

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
        ├─ AI        → ai-image-review 板块评审（含提示词改进建议）
        ├─ 风景/人物 → 对应板块评审（触发式双检复核 + 服务端规则）
        ├─ 其他      → 拒绝（route=rejected）
        └─ 疑似 AI（疑似AI率 ≥50%）→ 板块评审 + 【AI 疑似】提醒 + 提示词改进建议
```

LLM 层为 provider 适配器：默认 OpenAI 兼容接口（推荐通义 qwen-vl-max，DeepSeek / 智谱 GLM / 豆包亦可），`PHOTO_AGENT_PROVIDER=anthropic` 可切 Claude。

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
  "model": "qwen-vl-max",
  "usage": {"input_tokens": 13450, "output_tokens": 1652}
}
```

其他路由：AI 图 `route="ai"`（review 额外含 `prompt_suggestion` 提示词改进建议）；非风景/人物题材（如美食）`route="rejected"`（`review` 为 null）；疑似 AI（疑似AI率 ≥50%）附 `ai_suspicion`（warning 提醒 + prompt_suggestion 提示词改进建议）。

## 说明与声明

- **数据出境**：图片会随请求发往所配置的 LLM 服务商 API（评审的本质要求）。部署方使用自己的 API key 与账号，图片数据由服务商隐私政策约束。
- **鉴权（可选）**：设置 `PHOTO_AGENT_ACCESS_KEY` 后，`/api/review` 要求请求头 `X-API-Key`，校验失败返回 401；`/health`、`/skills` 保持开放。未设置 = 无鉴权（仅限本地开发）。内测/公开部署务必设置。
- **CORS（可选）**：设置 `PHOTO_AGENT_ALLOWED_ORIGINS`（逗号分隔）后允许对应来源的浏览器跨域调用。
- **自动压缩**：图片超过 `PHOTO_AGENT_MAX_IMAGE_MB`（默认 10MB）时自动压缩；超过 5 倍上限（默认 50MB）返回 413。元数据检测完成后，送 LLM 评审的图片统一压到长边 ≤3000px、≤2MB 以内（压缩在内存中完成，原图不会被修改；保留皮肤纹理细节供模型判断）。
- **并发上限**：`PHOTO_AGENT_MAX_CONCURRENT_REVIEWS`（默认 4）限制同时在途的评审请求；排队请求在拿到名额前不解析请求体，防止大图请求堆叠撑爆内存。
- **AI 检测边界**：元数据（C2PA/EXIF）命中是高置信信号；LLM 视觉判定输出的是置信度三档（ai / not_ai / uncertain），不是绝对结论。
- **评分体系**：人物/风景板块按 构图(3分)/光线(4分)/后期(3分) 评分（总分 10 分）；AI 图板块按 主体主题(3分)/特效场景(4分)/光效对比(3分) 评分（总分 10 分），另输出提示词改进建议。重大问题命中时所属维度封顶 1.0（展示「常规 X-Y（重大问题扣分）= Z」），满分须通过多层代码锁校验。
