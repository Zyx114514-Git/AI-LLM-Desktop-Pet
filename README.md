# 🐱 桌面宠物：眼球跟随 + AI对话 + 本地LLM + QLoRA微调

一个 Windows 透明桌面宠物（PyQt6）：猫耳少女会一直看着你的鼠标、偶尔眨眼，鼠标滑过耳朵切换表情，闲置时自动切到人形待机。支持 AI 对话（气泡呈现）、在线 API、本地开源 LLM 加载、QLoRA 微调、屏幕 OCR 感知。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qt](https://img.shields.io/badge/PyQt6-6.x-green)
![Platform](https://img.shields.io/badge/Windows-10%2B-lightblue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 👀 眼球跟随 | 眼球实时跟随鼠标，椭圆限位不穿出眼眶 |
| 😊 自然眨眼 | 每 2.5~4 秒随机眨眼一次 |
| 👂 耳朵交互 | 鼠标划过左/右耳切换表情 |
| 💤 待机表情 | 30 秒无操作自动切到人形待机 |
| 💬 AI 对话 | 漫画式气泡，可拖拽输入框 |
| 🧠 本地规则 | 基于 `knowledge/` 训练文本关键词匹配，离线可用 |
| 🌐 在线 API | 兼容 OpenAI 接口（智谱 GLM、DeepSeek 等） |
| 🤖 本地 LLM | 加载 HuggingFace 格式开源模型（Qwen/Llama 等） |
| 🔧 QLoRA 微调 | 4bit 量化 + LoRA，低显存微调自己的模型 |
| 📺 屏幕感知 | Windows 自带 OCR 识别屏幕内容作为对话上下文 |
| 📐 可调大小 | 右键菜单 15%~100% 缩放 |

---

## 📁 文件夹结构

```
desktop_pet/
├── pet.py                  # 主程序（入口）
├── llm_manager.py          # LLM 管理器：加载/推理/QLoRA微调
├── screen_utils.py         # 截图 + OCR 屏幕感知
├── config.json             # AI API 配置（可选）
├── requirements.txt        # pip 依赖
├── img/                    # 透明 PNG 素材
│   ├── idle_empty.png      #   身体+脸部底板（不含眼球）
│   ├── eye_left.png        #   左眼眼球
│   ├── eye_right.png       #   右眼眼球
│   ├── eye_lid_left.png    #   左眼闭眼眼睑
│   ├── eye_lid_right.png   #   右眼闭眼眼睑
│   ├── expr_left.png       #   左耳悬停表情
│   ├── expr_right.png      #   右耳悬停表情
│   └── expr_idle.png       #   30秒无操作的待机表情
├── knowledge/              # 训练文本（本地规则引擎，可自行添加 .txt）
│   └── example.txt
├── datasets/               # LLM 微调数据集（.jsonl/.json/.csv）
│   └── example.jsonl
└── finetune_output/        # QLoRA 微调输出（自动创建）
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/desktop_pet.git
cd desktop_pet
```

### 2. 安装依赖

```bash
# 核心依赖（桌宠 + 本地规则对话）
pip install PyQt6 Pillow

# 可选：在线 API 对话无需额外依赖（用标准库 urllib）

# 可选：本地 LLM + QLoRA 微调
pip install torch transformers peft bitsandbytes accelerate datasets trl

# 可选：屏幕 OCR（Windows 10+ 自带 OCR）
pip install winocr
```

完整依赖见 `requirements.txt`。

### 3. 运行

```bash
python pet.py
```

### 4. 操作方式

| 操作 | 效果 |
|------|------|
| **左键拖拽** | 移动桌宠 |
| **双击桌宠** | 弹出对话输入框 |
| **右键菜单** | 功能菜单 |
| **划过耳朵** | 切换表情 |

---

## 📖 使用说明

### AI 对话（三模式自动切换）

1. **本地规则引擎**（默认）：基于 `knowledge/` 文件夹的训练文本做关键词匹配，完全离线。
   - 格式：`问题|回答` 为问答对，无 `|` 的为纯知识/设定
2. **在线 API**：在 `config.json` 填写 API 配置后自动启用。
3. **本地 LLM**：加载开源模型后自动切换到 LLM 推理。

### config.json 配置示例

```json
{
  "base_url": "https://open.bigmodel.cn/api/paas/v4",
  "api_key": "你的API Key",
  "model": "glm-4-flash",
  "system_prompt": "你是一个可爱的猫娘桌宠，用中文简短回复，语气活泼可爱，每句话末尾加一个喵。"
}
```

- `api_key` 留空则使用本地规则引擎
- `system_prompt` 同时作用于在线 API 和本地 LLM 模式
- `knowledge/` 文件夹内容会自动追加到 system_prompt 作为背景设定

### 本地 LLM 模型

右键 → **LLM模型 → 加载模型…**，选择 HuggingFace 格式的模型目录。

**推荐 CPU 可跑的小模型：**
- Qwen2.5-0.5B-Instruct / Qwen2.5-1.5B-Instruct
- TinyLlama-1.1B-Chat

**模型下载**：从 [HuggingFace](https://huggingface.co) 或 [HF Mirror](https://hf-mirror.com) 下载整个模型文件夹。

### QLoRA 微调

1. 把数据集放进 `datasets/`（支持 JSONL / JSON / CSV）
2. 右键 → **LLM模型 → 开始微调…**
3. 选择数据集 → 设置轮数/学习率/LoRA rank → 开始
4. 微调完成后在 **加载微调适配器…** 中选择输出目录

**数据集格式（JSONL 推荐）：**

```jsonl
{"instruction": "你好", "input": "", "output": "你好呀喵"}
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀喵"}]}
{"prompt": "你是谁", "completion": "我是你的猫娘助手"}
```

### 屏幕感知（OCR）

右键菜单勾选 **"屏幕感知"**，对话时桌宠会先截图识别屏幕上的文字，作为上下文一并交给 AI 回答。依赖 Windows 10+ 自带 OCR（`pip install winocr`）。

---

## 🔧 关键参数（pet.py 顶部可调）

| 参数 | 含义 | 默认 |
|------|------|------|
| `SCALE` | 初始缩放比例 | `0.25` |
| `BLINK_MIN/MAX_INTERVAL` | 眨眼间隔（秒） | `2.5 / 4.0` |
| `IDLE_TIMEOUT` | 无操作多久切待机（秒） | `30` |
| `CHAT_BUBBLE_DURATION` | 气泡显示时长（毫秒） | `15000` |
| `LLM_MAX_NEW_TOKENS` | LLM 最大生成 token | `200` |
| `LLM_TEMPERATURE` | LLM 推理温度 | `0.7` |

---

## 📦 打包为 exe

> ⚠️ 含 LLM 依赖时建议用 `--onedir` 文件夹模式，单文件 `-F` 模式打包 torch 容易内存溢出。

```bash
pip install pyinstaller

# 含 LLM + OCR 的完整打包（onedir 模式）
pyinstaller -w -n "desktop_pet" ^
  --add-data "img;img" --add-data "knowledge;knowledge" --add-data "datasets;datasets" ^
  --collect-all torch --collect-all transformers --collect-all peft --collect-all bitsandbytes ^
  --hidden-import winocr ^
  --hidden-import winrt.windows.media.ocr ^
  --hidden-import winrt.windows.globalization ^
  --hidden-import winrt.windows.graphics.imaging ^
  --hidden-import winrt.windows.storage.streams ^
  --hidden-import winrt.windows.foundation ^
  pet.py --noconfirm
```

产物在 `dist/desktop_pet/` 文件夹，运行其中的 `desktop_pet.exe`。把 `config.json` 放在 exe 同级目录即可。

---

## 🛠️ 技术要点

- **透明窗口**：`Qt.FramelessWindowHint` + `WA_TranslucentBackground`，`paintEvent` 用 `drawPixmap` 原样叠加，不自动填充背景、不加边框阴影
- **分层绘制**：身体底板 → 左眼 → 右眼 → 眼睑，各层独立 QPixmap，不合并
- **眼球限位**：椭圆方程 `(dx/max_dx)² + (dy/max_dy)² ≤ 1`，保证眼珠不穿出眼眶
- **对话气泡**：独立无边框透明窗口，圆角白底 + 底部三角尾巴，自动换行
- **异步 LLM**：模型加载、推理、微调均在 `QThread` 后台线程，不阻塞 UI
- **依赖可选**：LLM/OCR 依赖未安装时，核心桌宠功能不受影响

---

## 📄 License

MIT
