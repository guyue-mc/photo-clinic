# PhotoClinic — 项目指令

## 图片评审流程（强制）

但凡用户向本 agent 提出图片评审请求（评审/鉴定/看图/给结果等），Claude **只负责转发**，不得自行看图评审：

1. **拿图**：优先使用用户给的路径；粘贴的附件若未带路径，请用户将图存到 `d:\测试图` 或提供路径。
2. **确保服务器在跑**：`curl -s -m 5 http://127.0.0.1:8000/health` 探活；未运行则后台启动：
   `.venv/Scripts/python.exe -m uvicorn photo_clinic.server:create_app --factory --host 0.0.0.0 --port 8000`
3. **转发**：将图片文件转 base64，`POST /api/review`，请求体 `{"image_base64": "<base64>", "media_type": "<MIME或省略>"}`；若返回 401，从 `.env` 的 `PHOTO_AGENT_ACCESS_KEY` 取密钥加 `X-API-Key` 头。
4. **回传**：把返回 JSON 按对应板块 skill 的「点评输出模板」转述输出（纯格式转换，不添加画面解读、不增删内容）：
   - 模板来源：`review.skill` 对应的 SKILL.md（`portrait-review` / `landscape-review` / `ai-image-review`）中「点评输出模板」段落；route=rejected（题材不属于风景/人物）时无点评，按排版 JSON 原样回传。
   - 遵循该模板下的「展示规则」：点评各段落只写评语与扣分点、不写子项小分；【评分】行不显示满分、只写分值本身与总分；直接输出模板，不加额外标题，不输出 route / 置信度 / usage 等内部字段。
   - 若返回含 `ai_suspicion`（仅疑似AI率 ≥50% 时出现），在【评分】行上方加一行：`【AI 疑似】<warning 内容>`；若 `ai_suspicion.prompt_suggestion` 非空，在【AI 疑似】行后追加该提示词改进建议（以「如果是AI生图」开头）。
5. **识别与评审由服务器完成**：服务器通过 provider 层调用配置的视觉模型（默认千问 `qwen3-vl-32b-instruct`，见 `.env.example` / `src/photo_clinic/config.py`）完成 AI 判定、题材分类与板块评审。

**禁止**：Claude 自行调用 `photo-review` / `ai-detection` / `subject-classify` / `landscape-review` / `portrait-review` 等 skill 直接评审图片；禁止自行描述画面内容做题材判断。本项目的 skill 文件仅供服务器读取（`skills_dir`），不供 Claude 在本会话执行。
