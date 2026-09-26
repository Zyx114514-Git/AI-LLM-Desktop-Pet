# -*- coding: utf-8 -*-
"""
桌面宠物：眼球跟随鼠标（PyQt6）
================================

目录结构（pet.py 与 img/ 同级）：
    desktop_pet/
    ├── pet.py                # 本程序
    ├── requirements.txt      # pip 依赖
    ├── README.md             # 使用/打包说明
    └── img/
        ├── idle_empty.png    # 身体+脸部底板（无眼球，透明背景）
        ├── eye_left.png      # 左眼眼球（透明背景）
        └── eye_right.png     # 右眼眼球（透明背景）

功能：
  1. 无边框 + 置顶 + 背景透明；左键拖拽移动桌宠；右键菜单仅“退出”。
  2. 分层绘制：第 1 层画身体底板，第 2/3 层分别叠加左右眼球。
  3. 眼球随屏幕鼠标坐标移动，且只在眼眶范围内限位偏移（椭圆限位），不穿出眼眶。
  4. 眼球素材为透明 PNG，直接原图叠加：不填充底色、不描边、不加阴影、不加任何
     渲染效果；图片不拉伸，保持原始长宽比（仅允许整体等比缩放）。
  5. 左右眼球各自独立配置“眼眶中心 / 最大偏移”，互相解耦。

坑点规避：
  - 启用 Qt.WA_TranslucentBackground 透明属性（否则窗口会被黑色填充）。
  - paintEvent 里只 drawPixmap，绝不调用 fillRect / 画边框 / 设置画笔。
  - 拖拽用 globalPosition() - 按下点偏移计算，避免坐标错位。
  - 眼球偏移经过椭圆方程归一化限位，保证任何方向都不穿出眼眶。
"""

import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime
import urllib.request

from PyQt6.QtCore import QPointF, QRectF, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QCursor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from screen_utils import get_screen_context, check_ocr_available
from llm_manager import (
    DEFAULT_FINETUNE_PARAMS,
    FinetuneWorker,
    GenerateWorker,
    LLMManager,
    ModelLoadWorker,
    check_dependencies,
    load_dataset_file,
)

# ==================== 可调配置 ====================

# 整体等比缩放（不改变长宽比，1.0 = 素材原始尺寸）。
# 素材为 1856x2048，默认 0.25 约为 464x512；随时可在右键菜单“调整大小”里改。
SCALE = 0.25

# 右键菜单“调整大小”可选的缩放档位（等比缩放，不改长宽比）
SCALE_OPTIONS = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]

# 鼠标离桌宠中心多远时眼球达到最大偏移（屏幕像素，越大越“迟钝”）
MAX_GAZE_DIST = 300

# 鼠标在桌宠中心附近的小死区（像素），避免眼球轻微抖动
DEAD_ZONE = 8

# ==================== 眨眼参数 ====================

BLINK_MIN_INTERVAL = 2.5   # 两次眨眼的最小间隔（秒）
BLINK_MAX_INTERVAL = 4.0   # 两次眨眼的最大间隔（秒）
BLINK_CLOSE_TIME = 0.07    # 闭眼时长（秒）
BLINK_OPEN_TIME = 0.07     # 睁眼时长（秒）
BLINK_HOLD_TIME = 0.05     # 完全闭合后的停留时长（秒）——一次眨眼全程约 0.19s，短促自然

# ==================== 耳朵表情切换 ====================
# 鼠标滑过左/右耳时，整图切换为对应表情（上传图转透明后存放在 img/ 下）；
# 鼠标离开耳朵后恢复分层渲染（眼球跟随 + 眨眼）。
# ear: (x0, y0, x1, y1) 耳朵热区，原图坐标系下的矩形（程序自动乘 SCALE）
EXPRESSIONS = [
    {"file": "expr_left.png",  "ear": (620, 175, 800, 480)},    # 左耳
    {"file": "expr_right.png", "ear": (1050, 175, 1230, 480)},  # 右耳
]
# 长时间无操作时显示的空闲表情
IDLE_EXPR_FILE = "expr_idle.png"
IDLE_TIMEOUT = 30  # 秒，无操作超过此时长自动切空闲表情

# ==================== AI对话配置 ====================
# 训练文本文件夹（与 pet.py 同级），用户可自行添加 .txt 文件
# 文件格式：每行一条；"问题|回答" 为问答对，无 | 的为纯知识/设定
KNOWLEDGE_DIR = "knowledge"
# API 配置文件（与 pet.py 同级），api_key 留空则用本地规则引擎
AI_CONFIG_FILE = "config.json"
# 对话气泡显示时长（毫秒），到期自动隐藏
CHAT_BUBBLE_DURATION = 15000
# 气泡最大宽度（像素），超出自动换行
CHAT_BUBBLE_MAX_WIDTH = 280

# ==================== LLM 开源模型 + QLoRA 微调配置 ====================
# 数据集文件夹（用户上传的微调数据集存放在这里）
DATASETS_DIR = "datasets"
# 微调输出文件夹（微调后的 LoRA 适配器保存在这里）
FINETUNE_OUTPUT_DIR = "finetune_output"
# LLM 推理最大生成 token 数
LLM_MAX_NEW_TOKENS = 200
# LLM 推理温度
LLM_TEMPERATURE = 0.7

# 眼球限位参数（单位：原图 1856x2048 坐标系下的像素，程序会自动乘 SCALE）
#   home        : 眼球在身体图上的“眼眶中心”，即静止时眼珠所在位置
#   max_dx      : 水平最大偏移（左右对称）
#   max_dy_up   : 垂直向上最大偏移
#   max_dy_down : 垂直向下最大偏移
# 数值来源：对素材实测——左眼眶白区宽约 106px（眼珠 64px），右眼眶窄约 75px 且内眼角
#   紧邻皮肤，故左右取相同的保守值 12px，保证两只眼珠同步移动且任何方向都不穿出眼眶；
#   垂直方向眼珠几乎贴满眼眶上下边，余量极小，因此垂直偏移设得很小（主要靠水平体现跟随）。
#   想更夸张可自行调大 max_dx，但注意右眼内眼角方向会先顶到眼眶边缘。
EYES = [
    {"file": "eye_left.png",  "home": (838, 608), "socket": (787, 559, 892, 652), "lid": "eye_lid_left.png",
     "max_dx": 12, "max_dy_up": 4, "max_dy_down": 1},
    {"file": "eye_right.png", "home": (1026, 604), "socket": (986, 555, 1065, 652), "lid": "eye_lid_right.png",
     "max_dx": 12, "max_dy_up": 4, "max_dy_down": 1},
]

# ==================== 工具函数 ====================


def resource_path(relative: str) -> str:
    """兼容 PyInstaller 打包：exe 运行时素材在 _MEIPASS 临时目录；源码运行时在脚本同目录。"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


def load_pixmap_raw(rel_path: str) -> QPixmap:
    """加载素材原始尺寸（不做任何缩放），便于运行时动态调整大小。"""
    pm = QPixmap(resource_path(rel_path))
    if pm.isNull():
        raise FileNotFoundError(f"找不到素材文件: {rel_path}")
    return pm


def scale_pixmap(pm: QPixmap, scale: float) -> QPixmap:
    """整体等比缩放（不改长宽比、不拉伸变形），scale==1.0 时原样返回。"""
    if scale == 1.0:
        return pm
    return pm.scaled(
        round(pm.width() * scale),
        round(pm.height() * scale),
        Qt.AspectRatioMode.KeepAspectRatio,  # 保持原始比例，禁止拉伸变形
        Qt.TransformationMode.SmoothTransformation,
    )


def app_dir() -> str:
    """程序所在目录（源码运行时=脚本目录；打包后=exe所在目录，用于读写配置和训练文本）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ==================== AI对话管理器 ====================

class AIChatManager:
    """
    AI对话管理器：双模式
      1. 本地规则引擎（默认）：基于 knowledge/ 文件夹的训练文本做关键词匹配，无需联网。
      2. 在线API（可选）：config.json 中填写 base_url + api_key 后，调用 OpenAI 兼容接口，
         训练文本作为 system prompt 背景知识。
    训练文本格式：每行一条；"问题|回答" 为问答对，无 | 的为纯知识/设定。
    """

    DEFAULT_REPLIES = [
        "嗯？你说什么？",
        "我在听哦~",
        "可以再说一遍吗？",
        "喵？",
        "（歪头看着你）",
        "人家没听懂啦~",
    ]

    def __init__(self):
        self.knowledge = []       # 纯知识/设定行
        self.qa_pairs = []        # 问答对 [(问题, 回答), ...]
        self.api_config = None    # {"base_url", "api_key", "model"}
        self.load_config()
        self.load_knowledge()

    # ---------- 配置与训练文本加载 ----------

    def load_config(self):
        """加载 config.json（优先 exe 同级目录，其次打包内置目录）。"""
        for base in (app_dir(), os.path.dirname(os.path.abspath(__file__))):
            path = os.path.join(base, AI_CONFIG_FILE)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.api_config = json.load(f)
                    return
                except (json.JSONDecodeError, OSError):
                    continue
        self.api_config = None

    def load_knowledge(self):
        """加载 knowledge/ 文件夹下所有 .txt 文件（优先 exe 同级目录，其次打包内置）。"""
        self.knowledge = []
        self.qa_pairs = []
        for base in (app_dir(), os.path.dirname(os.path.abspath(__file__))):
            kdir = os.path.join(base, KNOWLEDGE_DIR)
            if not os.path.exists(kdir):
                continue
            for fname in sorted(os.listdir(kdir)):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(kdir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "|" in line:
                                q, a = line.split("|", 1)
                                q, a = q.strip(), a.strip()
                                if q and a:
                                    self.qa_pairs.append((q, a))
                            else:
                                self.knowledge.append(line)
                except OSError:
                    continue
            if self.qa_pairs or self.knowledge:
                return  # 找到就不再找内置目录
        # 都没找到则创建空目录（exe同级）
        kdir = os.path.join(app_dir(), KNOWLEDGE_DIR)
        os.makedirs(kdir, exist_ok=True)

    def reload(self):
        """重新加载配置和训练文本（右键菜单调用）。"""
        self.load_config()
        self.load_knowledge()

    @property
    def api_enabled(self) -> bool:
        """是否启用了在线API（api_key 非空）。"""
        return bool(self.api_config and self.api_config.get("api_key"))

    # ---------- 获取回复 ----------

    def get_response(self, user_input: str, screen_context: str = "") -> str:
        """获取AI回复：优先在线API，失败回退本地规则引擎。"""
        if self.api_enabled:
            try:
                return self._call_api(user_input)
            except Exception:
                pass  # API失败时静默回退到本地规则
        return self._rule_based(user_input)

    def _detect_time_query(self, u: str):
        """识别时间/日期/星期查询，返回回复；不是时间问题返回 None。"""
        now = datetime.now()
        # 现在几点 / 时间
        if any(k in u for k in ("几点", "什么时间", "现在时间", "当前时间", "time")):
            return f"现在是 {now.strftime('%H:%M')} 啦~"
        # 今天几号 / 日期
        if any(k in u for k in ("几号", "什么日期", "今天日期", "当前日期", "date")):
            return f"今天是 {now.strftime('%Y年%m月%d日')}"
        # 星期几
        if any(k in u for k in ("星期", "周几", "礼拜")):
            weekdays = ["一", "二", "三", "四", "五", "六", "日"]
            return f"今天是星期{weekdays[now.weekday()]}"
        # 今天周几（没带星期关键词但问"周"）
        if "周" in u and any(k in u for k in ("今天", "现在", "这")):
            weekdays = ["一", "二", "三", "四", "五", "六", "日"]
            return f"今天是星期{weekdays[now.weekday()]}"
        return None
    def _rule_based(self, user_input: str, screen_context: str = "") -> str:
        """本地规则引擎：时间日期识别 → 问答对关键词匹配 → 屏幕内容检索 → 随机知识 → 默认回复。"""
        u = user_input.lower()
        u_words = set(u.replace("?", "").replace("？", "").replace("!", "").replace("！", "").split())

        # 0) 实时时间/日期/星期识别（优先级最高，直接返回真实时间）
        time_reply = self._detect_time_query(u)
        if time_reply:
            return time_reply

        # 1) 问答对匹配：用户输入与问题的关键词重合度最高者
        best, best_score = None, 0.0
        for q, a in self.qa_pairs:
            q_words = set(q.lower().replace("?", "").replace("？", "").split())
            if not q_words:
                continue
            common = q_words & u_words
            # 完全包含问题或问题包含用户输入，直接高分
            if q in u or u in q:
                score = 1.0
            else:
                score = len(common) / len(q_words)
            if score > best_score:
                best_score, best = score, a
        if best and best_score >= 0.3:
            return best

        # 2) 屏幕内容检索：在屏幕OCR文字中找与用户问题相关的行
        if screen_context:
            screen_lines = [ln.strip() for ln in screen_context.splitlines() if ln.strip()]
            relevant = []
            for line in screen_lines:
                lw = set(line.lower().split())
                common = lw & u_words
                if len(common) >= 1 and len(line) < 200:
                    relevant.append((len(common), line))
            if relevant:
                relevant.sort(reverse=True)
                top = relevant[0][1]
                if any(k in u for k in ("屏幕", "当前", "现在", "上面", "这里", "什么", "看看", "读一下", "识别", "内容")):
                    preview = screen_context.strip().replace(chr(10), " ")[:120]
                    return f"我看到屏幕上有：{preview}"
                return f"（看到屏幕内容）{top}"

        # 3) 随机返回一条知识（如果有）
        if self.knowledge and random.random() < 0.4:
            return random.choice(self.knowledge)

        # 4) 默认回复
        return random.choice(self.DEFAULT_REPLIES)

    def _call_api(self, user_input: str, screen_context: str = "") -> str:
        """调用 OpenAI 兼容 /chat/completions 接口（urllib，无额外依赖）。"""
        system_content = self.api_config.get("system_prompt") or "你是一个可爱的桌面宠物，用中文简短回复，语气活泼可爱。"
        if self.knowledge:
            system_content += "\n\n背景设定：\n" + "\n".join(self.knowledge[:30])
        if self.qa_pairs:
            system_content += "\n\n对话示例：\n" + "\n".join(
                f"用户：{q}\n宠物：{a}" for q, a in self.qa_pairs[:15]
            )

        payload = json.dumps({
            "model": self.api_config.get("model", "gpt-3.5-turbo"),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input},
            ],
            "max_tokens": 150,
            "temperature": 0.8,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_config["base_url"].rstrip("/") + "/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_config['api_key']}",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


# ==================== AI工作线程（避免阻塞UI） ====================

class AIWorker(QThread):
    """后台线程调用 AI，完成后通过信号返回结果。"""
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, manager: AIChatManager, user_input: str, screen_context: str = ""):
        super().__init__()
        self._manager = manager
        self._input = user_input
        self._screen = screen_context

    def run(self):
        try:
            reply = self._manager.get_response(self._input, self._screen)
            self.finished.emit(reply)
        except Exception as exc:
            self.failed.emit(str(exc))


# ==================== 自定义对话输入框 ====================

class ChatInputDialog(QDialog):
    """
    可爱风格的对话输入框：圆角奶油白底 + 粉色描边 + 圆角按钮。
    回车发送，ESC 取消，和桌宠气泡风格统一。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("和宠物说话")
        # 无边框 + 置顶 + 任务栏不显示
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(320, 90)

        # 外层容器（圆角背景）
        container = QWidget(self)
        container.setObjectName("inputContainer")
        container.setGeometry(0, 0, 320, 90)
        container.setStyleSheet(
            """
            #inputContainer {
                background-color: #fff8f5;
                border: 2px solid #ffb6c1;
                border-radius: 18px;
            }
            QLabel#titleLabel {
                color: #e07a8a;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
            }
            QLineEdit {
                background-color: #ffffff;
                border: 2px solid #ffc0cb;
                border-radius: 12px;
                padding: 8px 12px;
                font-size: 14px;
                color: #5a3e4a;
                selection-background-color: #ffb6c1;
            }
            QLineEdit:focus {
                border: 2px solid #ff8fa3;
            }
            QPushButton {
                border-radius: 12px;
                padding: 6px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton#sendBtn {
                background-color: #ff8fa3;
                color: white;
                border: none;
            }
            QPushButton#sendBtn:hover {
                background-color: #ff6b81;
            }
            QPushButton#cancelBtn {
                background-color: #ffe4e9;
                color: #e07a8a;
                border: none;
            }
            QPushButton#cancelBtn:hover {
                background-color: #ffd0d9;
            }
            """
        )

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # 输入框
        self.edit = QLineEdit(container)
        self.edit.setPlaceholderText("输入你想说的话...")
        self.edit.returnPressed.connect(self.accept)
        layout.addWidget(self.edit)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.cancel_btn = QPushButton("取消", container)
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.clicked.connect(self.reject)
        self.send_btn = QPushButton("发送", container)
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.send_btn)
        layout.addLayout(btn_row)

        self.edit.setFocus()

    def text(self) -> str:
        return self.edit.text().strip()

    # ---- 无边框窗口拖动 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and hasattr(self, "_drag_pos"):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

# ==================== 对话气泡 ====================

class ChatBubble(QWidget):
    """
    精致对话气泡：圆角奶油白底 + 柔和描边 + 圆润尾巴，浅粉色调。
    显示在桌宠上方居中，自动换行，到时自动隐藏。
    """

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._text = ""
        self._lines = []
        self._padding_x = 16
        self._padding_y = 12
        self._tail_h = 12
        self._radius = 16
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    def show_text(self, text: str, duration: int = CHAT_BUBBLE_DURATION):
        """显示气泡文本，duration 毫秒后自动隐藏。"""
        self._text = text
        self._wrap_text()
        self._resize_to_content()
        self.show()
        self.raise_()
        self._hide_timer.start(duration)

    def _bubble_font(self):
        font = QFont("Microsoft YaHei", 11)
        font.setPointSize(11)
        return font

    def _wrap_text(self):
        """按最大宽度自动换行。"""
        font = self._bubble_font()
        metrics = QFontMetrics(font)
        self._lines = []
        for paragraph in self._text.split("\n"):
            if metrics.horizontalAdvance(paragraph) <= CHAT_BUBBLE_MAX_WIDTH:
                self._lines.append(paragraph)
                continue
            current = ""
            for ch in paragraph:
                if metrics.horizontalAdvance(current + ch) > CHAT_BUBBLE_MAX_WIDTH and current:
                    self._lines.append(current)
                    current = ch
                else:
                    current += ch
            if current:
                self._lines.append(current)

    def _resize_to_content(self):
        """根据文本行数计算气泡尺寸。"""
        font = self._bubble_font()
        metrics = QFontMetrics(font)
        line_h = metrics.height()
        text_w = min(
            CHAT_BUBBLE_MAX_WIDTH,
            max((metrics.horizontalAdvance(l) for l in self._lines), default=100),
        )
        text_h = line_h * len(self._lines)
        self._text_w = text_w
        self._text_h = text_h
        self._line_h = line_h
        self.setFixedSize(
            text_w + self._padding_x * 2,
            text_h + self._padding_y * 2 + self._tail_h,
        )

    def paintEvent(self, _event):
        """绘制气泡：柔和渐变圆角主体 + 圆润尾巴 + 精致文字。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # 气泡主体区域
        body_rect = QRectF(0, 0, self.width(), self.height() - self._tail_h)

        # 柔和阴影（底部偏移的半透明圆角矩形）
        shadow_rect = QRectF(2, 3, body_rect.width(), body_rect.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(180, 140, 160, 30))
        painter.drawRoundedRect(shadow_rect, self._radius, self._radius)

        # 主体填充：奶油白到淡粉的柔和渐变
        from PyQt6.QtGui import QLinearGradient
        grad = QLinearGradient(0, 0, 0, body_rect.height())
        grad.setColorAt(0, QColor(255, 252, 253, 252))
        grad.setColorAt(1, QColor(255, 240, 245, 252))
        painter.setBrush(grad)
        painter.setPen(QPen(QColor(240, 180, 200, 180), 1.2))
        painter.drawRoundedRect(body_rect, self._radius, self._radius)

        # 圆润尾巴（贝塞尔曲线，指向下方桌宠）
        tail_x = self.width() // 2
        tail_top = self.height() - self._tail_h
        tail = QPainterPath()
        tail.moveTo(tail_x - 9, tail_top + 1)
        # 左侧曲线
        tail.quadTo(tail_x - 8, tail_top + 6, tail_x - 3, tail_top + 9)
        tail.lineTo(tail_x, self.height())
        # 右侧曲线
        tail.lineTo(tail_x + 3, tail_top + 9)
        tail.quadTo(tail_x + 8, tail_top + 6, tail_x + 9, tail_top + 1)
        tail.closeSubpath()
        painter.fillPath(tail, QColor(255, 240, 245, 252))

        # 尾巴上描边（只画外侧两条线）
        painter.setPen(QPen(QColor(240, 180, 200, 180), 1.2))
        painter.drawLine(tail_x - 9, tail_top + 1, tail_x - 3, tail_top + 9)
        painter.drawLine(tail_x - 3, tail_top + 9, tail_x, self.height())
        painter.drawLine(tail_x, self.height(), tail_x + 3, tail_top + 9)
        painter.drawLine(tail_x + 3, tail_top + 9, tail_x + 9, tail_top + 1)

        # 文字（深棕灰，柔和不刺眼）
        font = self._bubble_font()
        painter.setFont(font)
        painter.setPen(QColor(70, 50, 55))
        y = self._padding_y + QFontMetrics(font).ascent()
        for line in self._lines:
            painter.drawText(self._padding_x, int(y), line)
            y += self._line_h

        painter.end()


# ==================== 主窗口 ====================


class PetWindow(QWidget):
    def __init__(self):
        super().__init__()

        # ---- 窗口基础设置：无边框 + 置顶 + 工具窗口(不进任务栏) ----
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        # 关键：启用透明背景，否则窗口区域会被系统用不透明底色填充
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # 当前生效的缩放（初始取模块默认 SCALE）
        self.scale = SCALE

        # ---- 加载原始尺寸素材（含身体底板与左右眼球），缩放由 _apply_scale 统一处理 ----
        self._body_native = load_pixmap_raw(os.path.join("img", "idle_empty.png"))
        self._eye_native = [load_pixmap_raw(os.path.join("img", cfg["file"])) for cfg in EYES]

        # 按当前缩放初始化窗口尺寸与眼球几何
        self._apply_scale()

        # ---- 拖拽状态 ----
        self._drag_offset = None  # 按下时 鼠标全局坐标 - 窗口左上角，用于拖拽不偏移

        # ---- 眨眼状态机 ----
        self._blink_state = "waiting"        # waiting→closing→closed→opening→waiting
        self._blink_progress = 0.0           # 0=完全睁开，1=完全闭合
        self._blink_timer = random.uniform(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL)  # 距下次眨眼/停留结束的秒数
        self._last_tick = time.monotonic()   # 上一帧时间戳（用于计算真实 dt）

        # ---- 耳朵表情切换状态 ----
        self._expr_index = None  # None=正常分层渲染；0/1=当前悬停左/右耳，整图显示对应表情

        # ---- 空闲表情：长时间无操作自动切换 ----
        self._is_idle = False
        self._last_active = time.monotonic()  # 最后一次鼠标活动时间
        self.idle_pix = None

        # ---- AI对话 ----
        self.ai_manager = AIChatManager()
        self.bubble = ChatBubble()
        self._ai_worker = None  # 当前运行的AI工作线程

        # ---- LLM 开源模型 + QLoRA 微调 ----
        self.llm = LLMManager()
        self._llm_load_worker = None
        self._llm_gen_worker = None
        self._finetune_worker = None
        self._llm_deps_ok, self._llm_deps_missing = check_dependencies()

        # ---- 屏幕感知（截图 + OCR）----
        self._screen_perception = False
        self._ocr_ok, self._ocr_engine = check_ocr_available()

        # ---- 定时轮询（约 60fps）：驱动眼球跟随 + 眨眼动画 ----
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    # ---------- 帧推进 ----------

    def _tick(self):
        now = time.monotonic()
        dt = min(now - self._last_tick, 0.1)  # 窗口切换/卡顿后限制 dt，避免动画跳变
        self._last_tick = now
        self._check_idle(now)
        self._update_eye_offsets()
        self._update_expression()
        self._update_blink(dt)

    # ---------- 空闲表情 ----------

    def _check_idle(self, now: float):
        """检查是否进入/退出空闲状态。"""
        if self.idle_pix is None:
            return
        if not self._is_idle:
            # 超过 IDLE_TIMEOUT 秒无操作 → 进入空闲
            if now - self._last_active > IDLE_TIMEOUT:
                self._is_idle = True
                self._expr_index = None  # 空闲时不响应耳朵热区
                self.update()
        else:
            # 空闲中：鼠标移到桌宠上或有点击 → 退出空闲
            pos = QCursor.pos() - self.pos()
            if self.rect().contains(QPointF(pos).toPoint()):
                self._is_idle = False
                self._last_active = now
                self.update()

    def _wake(self):
        """用户有操作时调用：重置空闲计时并退出空闲状态。"""
        self._last_active = time.monotonic()
        if self._is_idle:
            self._is_idle = False
            self.update()

    # ---------- 耳朵表情切换 ----------

    def _update_expression(self):
        """
        检测鼠标是否悬停在左/右耳热区（窗口坐标）：
        悬停 → 记录对应表情索引，paintEvent 整图切换；离开 → 恢复 None。
        """
        if not self.expressions:
            return
        pos = QCursor.pos() - self.pos()  # 全局鼠标坐标 → 窗口坐标
        idx = None
        for i, ex in enumerate(self.expressions):
            if ex["rect"].contains(QPointF(pos)):
                idx = i
                break
        if idx != self._expr_index:
            self._expr_index = idx
            self.update()

    # ---------- 动态调整大小 ----------

    def _apply_scale(self):
        """按 self.scale 重新生成：身体图、眼球图、窗口尺寸、眼眶中心与最大偏移。"""
        # 第 1 层：身体底板（等比缩放，不改长宽比）
        self.body_pix = scale_pixmap(self._body_native, self.scale)
        self.setFixedSize(self.body_pix.size())  # 窗口尺寸 = 身体图尺寸

        # 第 2/3 层：左右眼球（透明 PNG 原样等比缩放，不做任何渲染处理）
        self.eyes = []
        for cfg, pix_native in zip(EYES, self._eye_native):
            self.eyes.append({
                "pix": scale_pixmap(pix_native, self.scale),  # 眼球整图（透明区域随图平移）
                "home": QPointF(cfg["home"][0] * self.scale, cfg["home"][1] * self.scale),  # 眼眶中心(窗口坐标)
                "max_dx": cfg["max_dx"] * self.scale,      # 水平最大偏移(窗口坐标)
                "max_dy_up": cfg["max_dy_up"] * self.scale,      # 垂直向上最大偏移
                "max_dy_down": cfg["max_dy_down"] * self.scale,  # 垂直向下最大偏移
                "offset": QPointF(0.0, 0.0),                # 当前绘制偏移(dx, dy)
            })

        # 第 4 层：闭眼眼睑（仅眨眼时绘制，用于闭眼扫过动画）
        self.lids = []
        for cfg in EYES:
            lid_pix = scale_pixmap(load_pixmap_raw(os.path.join("img", cfg["lid"])), self.scale)
            self.lids.append({
                "pix": lid_pix,
                "top": cfg["socket"][1] * self.scale,     # 眼眶顶部 y(窗口坐标)，裁剪扫过动画的起点
                "bottom": cfg["socket"][3] * self.scale,  # 眼眶底部 y，裁剪扫过动画的终点
            })

        # 表情图层：鼠标悬停在耳上时整图切换显示
        self.expressions = []
        for cfg in EXPRESSIONS:
            expr_pix = scale_pixmap(load_pixmap_raw(os.path.join("img", cfg["file"])), self.scale)
            x0, y0, x1, y1 = cfg["ear"]
            self.expressions.append({
                "pix": expr_pix,
                "rect": QRectF(x0 * self.scale, y0 * self.scale,
                               (x1 - x0) * self.scale, (y1 - y0) * self.scale),  # 耳朵热区(窗口坐标)
            })
        # 空闲表情图（用 resource_path 兼容打包后的临时目录）
        idle_path = os.path.join("img", IDLE_EXPR_FILE)
        if os.path.exists(resource_path(idle_path)):
            self.idle_pix = scale_pixmap(load_pixmap_raw(idle_path), self.scale)
        else:
            self.idle_pix = None
        self.update()

    def set_scale(self, value: float):
        """运行时调整桌宠整体大小（保持窗口中心位置不变）。"""
        if abs(value - self.scale) < 1e-9:
            return
        center = self.frameGeometry().center()  # 记录缩放前的窗口中心
        self.scale = value
        self._apply_scale()
        # 缩放后把窗口中心对齐回原位置，避免大小变化时窗口跳走
        self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)

    # ---------- 眼球跟随逻辑 ----------

    def _update_eye_offsets(self):
        """根据屏幕鼠标坐标计算左右眼球偏移，并做眼眶椭圆限位。"""
        mouse = QCursor.pos()  # 屏幕坐标
        center = self.frameGeometry().center()  # 桌宠窗口中心的屏幕坐标
        vec = QPointF(mouse - center)  # 鼠标相对窗口中心的向量

        # 死区：鼠标离中心很近时眼球归位，避免细微抖动
        if math.hypot(vec.x(), vec.y()) < DEAD_ZONE:
            self._apply_offsets(QPointF(0.0, 0.0), QPointF(0.0, 0.0))
            return

        changed = False
        for eye in self.eyes:
            # 1) 将向量归一化到 [-1, 1]：MAX_GAZE_DIST 之外即达到最大偏移
            nx = max(-1.0, min(1.0, vec.x() / MAX_GAZE_DIST))
            ny = max(-1.0, min(1.0, vec.y() / MAX_GAZE_DIST))

            # 2) 向量长度截断到 1，保证方向不变、只缩长度（等效于方形区域裁剪）
            length = math.hypot(nx, ny)
            if length > 1.0:
                nx /= length
                ny /= length

            # 3) 映射到眼眶椭圆半轴：水平 ±max_dx，垂直按方向取 max_dy_up / max_dy_down
            dy_axis = eye["max_dy_up"] if ny < 0 else eye["max_dy_down"]
            dx = nx * eye["max_dx"]
            dy = ny * dy_axis

            # 4) 椭圆限位（核心约束）：(dx/max_dx)^2 + (dy/dy_axis)^2 <= 1，
            #    确保眼珠在眼眶椭圆内，任何方向都不会穿出
            ratio = math.hypot(dx / eye["max_dx"], dy / dy_axis)
            if ratio > 1.0:
                dx /= ratio
                dy /= ratio

            new_offset = QPointF(dx, dy)
            if (new_offset - eye["offset"]).manhattanLength() > 0.2:
                changed = True
                eye["offset"] = new_offset

        # 只有偏移发生变化才重绘，省 CPU
        if changed:
            self.update()

    # ---------- 眨眼动画 ----------

    def _update_blink(self, dt: float):
        """
        眨眼状态机：waiting（随机等待 2.5~4s）→ closing（闭眼，70ms）
        → closed（停留 50ms）→ opening（睁眼，70ms）→ waiting。
        全程约 0.19s，短促自然。
        """
        repaint = False
        if self._blink_state == "waiting":
            self._blink_timer -= dt
            if self._blink_timer <= 0:
                self._blink_state = "closing"
                self._blink_progress = 0.0
                repaint = True
        elif self._blink_state == "closing":
            self._blink_progress += dt / BLINK_CLOSE_TIME
            if self._blink_progress >= 1.0:
                self._blink_progress = 1.0
                self._blink_state = "closed"
                self._blink_timer = BLINK_HOLD_TIME
            repaint = True
        elif self._blink_state == "closed":
            self._blink_timer -= dt
            if self._blink_timer <= 0:
                self._blink_state = "opening"
            repaint = True
        elif self._blink_state == "opening":
            self._blink_progress -= dt / BLINK_OPEN_TIME
            if self._blink_progress <= 0.0:
                self._blink_progress = 0.0
                self._blink_state = "waiting"
                self._blink_timer = random.uniform(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL)
            repaint = True

        if repaint:
            self.update()

    def _apply_offsets(self, left: QPointF, right: QPointF):
        """把两个眼球偏移直接写入（供测试/内部使用）；仅当偏移变化时才重绘，省 CPU。"""
        changed = False
        for eye, off in zip(self.eyes, (left, right)):
            if (off - eye["offset"]).manhattanLength() > 0.2:
                changed = True
            eye["offset"] = off
        if changed:
            self.update()

    # ---------- 分层绘制 ----------

    def paintEvent(self, _event):
        """
        分层绘制（顺序即图层顺序，绝不合并成一张图）：
          第 1 层：身体底板  drawPixmap(0, 0)
          第 2 层：左眼球    drawPixmap(dx, dy) —— 整图平移，透明区域随之移动
          第 3 层：右眼球    drawPixmap(dx, dy)
        透明窗口 + 透明 PNG 原样叠加：不填充底色、不描边、不加阴影。
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # ---- 空闲表情：长时间无操作，整图显示 idle 表情 ----
        if self._is_idle and self.idle_pix is not None:
            painter.drawPixmap(0, 0, self.idle_pix)
            painter.end()
            return

        # ---- 表情模式：鼠标悬停在左/右耳上，整图显示上传的表情 ----
        # 此时不绘制身体/眼球/眼睑（表情图本身已是完整全身闭眼微笑，无需叠加），
        # 保持原图原样、等比缩放、不拉伸。
        if self._expr_index is not None:
            painter.drawPixmap(0, 0, self.expressions[self._expr_index]["pix"])
            painter.end()
            return

        # ---- 第 1 层：身体 + 脸部底板（眼眶留白） ----
        painter.drawPixmap(0, 0, self.body_pix)

        # ---- 第 2、3 层：左右眼球（透明素材原样叠加） ----
        for eye in self.eyes:
            off = eye["offset"]
            painter.drawPixmap(QPointF(off.x(), off.y()), eye["pix"])

        # ---- 第 4 层：闭眼眼睑（仅眨眼时绘制） ----
        # 用垂直裁剪实现“眼皮从上往下扫”的闭眼/睁眼动作：
        #   progress 0→1：裁剪线从眼眶顶部扫到眼眶底部（闭眼）
        #   progress 1→0：裁剪线缩回眼眶顶部（睁眼）
        # 眼睑 PNG 与身体同坐标系，直接叠在眼眶上，不添加任何描边/阴影效果。
        if self._blink_progress > 0.0:
            p = min(self._blink_progress, 1.0)
            for lid in self.lids:
                clip_bottom = lid["top"] + (lid["bottom"] - lid["top"]) * p
                painter.save()
                painter.setClipRect(0, 0, lid["pix"].width(), int(clip_bottom))
                painter.drawPixmap(0, 0, lid["pix"])
                painter.restore()

        painter.end()

    # ---------- AI对话 ----------

    def show_chat_input(self):
        """弹出可爱风格输入框，用户输入后启动AI对话。"""
        dlg = ChatInputDialog(self)
        # 定位到桌宠旁边（右上角）
        gp = self.geometry()
        dlg.move(gp.right() + 20, gp.top() - 80)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.text():
            self.start_chat(dlg.text())

    def start_chat(self, user_input: str):
        """启动AI对话：显示"思考中"气泡，后台线程获取回复。"""
        # 屏幕感知：如果开关开启，先截图 OCR 获取屏幕内容
        screen_context = ""
        if self._screen_perception:
            try:
                self.bubble.show_text("（识别屏幕中…）", 5000)
                self._position_bubble()
                screen_context = get_screen_context()
            except Exception:
                screen_context = ""

        # 如果 LLM 模型已加载，优先使用 LLM 推理
        if self.llm.is_loaded:
            adapter_info = f"（微调适配器）" if self.llm.adapter_path else ""
            screen_tag = " +屏幕" if screen_context else ""
            self.bubble.show_text(f"（LLM{adapter_info}{screen_tag}）思考中…", 30000)
            self._position_bubble()
            if self._llm_gen_worker and self._llm_gen_worker.isRunning():
                self._llm_gen_worker.wait()
            self._llm_gen_worker = GenerateWorker(self.llm, user_input, LLM_MAX_NEW_TOKENS, screen_context=screen_context)
            self._llm_gen_worker.finished_ok.connect(self._on_ai_reply)
            self._llm_gen_worker.failed.connect(self._on_llm_failed)
            self._llm_gen_worker.start()
            return

        # 否则使用原 AI 管理器（本地规则 / 在线API）
        mode = "在线AI" if self.ai_manager.api_enabled else "本地规则"
        screen_tag = " +屏幕" if screen_context else ""
        self.bubble.show_text(f"（{mode}{screen_tag}）思考中…", 30000)
        self._position_bubble()

        if self._ai_worker and self._ai_worker.isRunning():
            self._ai_worker.wait()

        self._ai_worker = AIWorker(self.ai_manager, user_input, screen_context=screen_context)
        self._ai_worker.finished.connect(self._on_ai_reply)
        self._ai_worker.failed.connect(self._on_ai_failed)
        self._ai_worker.start()

    def _on_ai_reply(self, text: str):
        """AI回复到达：统一加"喵"尾音，显示在气泡中。"""
        text = (text or "").strip()
        # 去掉末尾标点空白，避免出现"。喵""！喵"
        while text and text[-1] in "。！？，、；：.!?~～ \n":
            text = text[:-1]
        if not text.endswith("喵"):
            text = text + "喵"
        self.bubble.show_text(text, CHAT_BUBBLE_DURATION)
        self._position_bubble()

    def _on_ai_failed(self, err: str):
        """AI调用失败：显示错误提示。"""
        self.bubble.show_text(f"（出错了：{err[:30]}）", 5000)
        self._position_bubble()

    def _on_llm_failed(self, err: str):
        """LLM推理失败：显示错误提示。"""
        self.bubble.show_text(f"（LLM出错：{err[:40]}）", 6000)
        self._position_bubble()

    def _toggle_screen_perception(self, checked):
        """切换屏幕感知开关。"""
        self._screen_perception = checked
        if checked:
            self.bubble.show_text("屏幕感知已开启\n对话时我会先看看你的屏幕~", 5000)
        else:
            self.bubble.show_text("屏幕感知已关闭", 3000)
        self._position_bubble()

    # ---------- LLM 模型管理 ----------

    def load_llm_model(self):
        """加载本地开源模型（HuggingFace 格式目录）。"""
        if not self._llm_deps_ok:
            QMessageBox.warning(
                self, "缺少依赖",
                "LLM 功能需要安装以下依赖：\n\n" + "\n".join(self._llm_deps_missing) +
                "\n\n安装命令：pip install torch transformers peft bitsandbytes accelerate datasets trl"
            )
            return

        model_path = QFileDialog.getExistingDirectory(self, "选择模型目录（HuggingFace 格式）")
        if not model_path:
            return

        # 检查是否为有效模型目录
        if not os.path.exists(os.path.join(model_path, "config.json")):
            QMessageBox.warning(self, "无效模型", "所选目录不是有效的 HuggingFace 模型目录（缺少 config.json）。")
            return

        self.bubble.show_text("正在加载模型，请稍候…", 60000)
        self._position_bubble()

        if self._llm_load_worker and self._llm_load_worker.isRunning():
            self._llm_load_worker.wait()

        self._llm_load_worker = ModelLoadWorker(self.llm, model_path, load_in_4bit=True)
        self._llm_load_worker.finished_ok.connect(self._on_llm_loaded)
        self._llm_load_worker.failed.connect(self._on_llm_load_failed)
        self._llm_load_worker.start()

    def _on_llm_loaded(self, model_path: str):
        """模型加载完成。"""
        # 注入 system_prompt：从 config.json 读取 + knowledge 背景设定
        sp = ""
        if self.ai_manager.api_config:
            sp = self.ai_manager.api_config.get("system_prompt", "")
        if self.ai_manager.knowledge:
            sp = (sp + "\n\n背景设定：\n" + "\n".join(self.ai_manager.knowledge[:30])).strip()
        self.llm.system_prompt = sp
        name = os.path.basename(model_path)
        self.bubble.show_text(f"模型已加载：{name}\n设备：{self.llm.device}\n现在可以对话了~", 8000)
        self._position_bubble()

    def _on_llm_load_failed(self, err: str):
        """模型加载失败。"""
        self.bubble.show_text(f"模型加载失败：{err[:50]}", 8000)
        self._position_bubble()

    def unload_llm_model(self):
        """卸载模型，释放显存。"""
        if not self.llm.is_loaded:
            return
        self.llm.unload_model()
        self.bubble.show_text("模型已卸载，显存已释放", 4000)
        self._position_bubble()

    # ---------- 数据集管理 ----------

    def upload_dataset(self):
        """上传数据集文件（复制到 datasets/ 文件夹）。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集文件", "",
            "数据集文件 (*.jsonl *.json *.csv);;所有文件 (*.*)"
        )
        if not file_path:
            return

        datasets_dir = os.path.join(app_dir(), DATASETS_DIR)
        os.makedirs(datasets_dir, exist_ok=True)

        dest = os.path.join(datasets_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dest)

        # 验证数据集格式
        try:
            samples = load_dataset_file(dest)
            self.bubble.show_text(
                f"数据集已导入：{os.path.basename(file_path)}\n有效样本：{len(samples)} 条",
                6000,
            )
        except Exception as e:
            self.bubble.show_text(f"数据集格式有误：{str(e)[:40]}", 6000)
        self._position_bubble()

    def open_datasets_dir(self):
        """打开数据集文件夹。"""
        d = os.path.join(app_dir(), DATASETS_DIR)
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    # ---------- QLoRA 微调 ----------

    def start_finetune(self):
        """启动 QLoRA 微调。"""
        if not self._llm_deps_ok:
            QMessageBox.warning(
                self, "缺少依赖",
                "微调需要安装以下依赖：\n\n" + "\n".join(self._llm_deps_missing) +
                "\n\n安装命令：pip install torch transformers peft bitsandbytes accelerate datasets trl"
            )
            return

        if not self.llm.model_path:
            # 让用户选择模型
            model_path = QFileDialog.getExistingDirectory(self, "选择基础模型目录")
            if not model_path:
                return
            self.llm.model_path = model_path
        else:
            model_path = self.llm.model_path

        # 列出 datasets/ 下的文件，让用户选择
        datasets_dir = os.path.join(app_dir(), DATASETS_DIR)
        os.makedirs(datasets_dir, exist_ok=True)
        dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith((".jsonl", ".json", ".csv"))]
        if not dataset_files:
            QMessageBox.information(self, "无数据集", f"数据集文件夹为空，请先在「{datasets_dir}」中放入数据集文件。")
            return

        dataset_name, ok = QInputDialog.getItem(
            self, "选择数据集", "请选择用于微调的数据集：", dataset_files, 0, False
        )
        if not ok:
            return
        dataset_path = os.path.join(datasets_dir, dataset_name)

        # 配置微调参数
        epochs, ok = QInputDialog.getInt(self, "微调参数", "训练轮数 (epochs)：", 3, 1, 20)
        if not ok:
            return
        lr, ok = QInputDialog.getDouble(self, "微调参数", "学习率：", 2e-4, 1e-6, 1e-2, 6)
        if not ok:
            return
        lora_r, ok = QInputDialog.getInt(self, "微调参数", "LoRA rank (r)：", 8, 1, 64)
        if not ok:
            return

        params = {
            **DEFAULT_FINETUNE_PARAMS,
            "num_train_epochs": epochs,
            "learning_rate": lr,
            "lora_r": lora_r,
        }

        # 输出目录
        output_dir = os.path.join(
            app_dir(), FINETUNE_OUTPUT_DIR,
            f"{os.path.basename(model_path)}_{int(time.time())}"
        )

        # 确认
        reply = QMessageBox.question(
            self, "确认微调",
            f"即将开始 QLoRA 微调：\n\n"
            f"模型：{os.path.basename(model_path)}\n"
            f"数据集：{dataset_name}\n"
            f"轮数：{epochs}\n"
            f"学习率：{lr}\n"
            f"LoRA rank：{lora_r}\n"
            f"输出：{output_dir}\n\n"
            f"注意：微调需要 GPU 显存，过程中程序可能变慢。确定开始吗？",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.bubble.show_text("开始 QLoRA 微调…\n（进度会在这里显示）", 60000)
        self._position_bubble()

        if self._finetune_worker and self._finetune_worker.isRunning():
            self._finetune_worker.wait()

        self._finetune_worker = FinetuneWorker(self.llm, model_path, dataset_path, output_dir, params)
        self._finetune_worker.progress.connect(self._on_finetune_progress)
        self._finetune_worker.finished_ok.connect(self._on_finetune_done)
        self._finetune_worker.failed.connect(self._on_finetune_failed)
        self._finetune_worker.start()

    def _on_finetune_progress(self, step: int, total: int, msg: str):
        """微调进度更新。"""
        if total > 0 and step >= 0:
            pct = min(100, int(step / total * 100))
            self.bubble.show_text(f"微调中… {pct}%\n{msg}", 60000)
        else:
            self.bubble.show_text(f"微调中…\n{msg}", 60000)
        self._position_bubble()

    def _on_finetune_done(self, output_dir: str):
        """微调完成。"""
        self.bubble.show_text(f"微调完成！\n适配器：{output_dir}\n可在菜单中加载使用", 15000)
        self._position_bubble()

    def _on_finetune_failed(self, err: str):
        """微调失败。"""
        self.bubble.show_text(f"微调失败：{err[:60]}", 10000)
        self._position_bubble()

    # ---------- 微调适配器管理 ----------

    def load_adapter(self):
        """加载微调后的 LoRA 适配器。"""
        if not self.llm.is_loaded:
            QMessageBox.warning(self, "模型未加载", "请先加载基础模型，再加载微调适配器。")
            return

        adapter_path = QFileDialog.getExistingDirectory(self, "选择微调适配器目录（包含 adapter_config.json）")
        if not adapter_path:
            return

        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            QMessageBox.warning(self, "无效适配器", "所选目录不是有效的 LoRA 适配器目录（缺少 adapter_config.json）。")
            return

        try:
            self.llm.load_adapter(adapter_path)
            self.bubble.show_text(f"已加载微调适配器：{os.path.basename(adapter_path)}", 6000)
        except Exception as e:
            self.bubble.show_text(f"加载适配器失败：{str(e)[:40]}", 6000)
        self._position_bubble()

    def unload_adapter(self):
        """卸载微调适配器，恢复基础模型。"""
        if not self.llm.adapter_path:
            return
        self.llm.model = self.llm.model.unload()
        self.llm.adapter_path = None
        self.bubble.show_text("已卸载微调适配器，恢复基础模型", 4000)
        self._position_bubble()

    def open_finetune_dir(self):
        """打开微调输出文件夹。"""
        d = os.path.join(app_dir(), FINETUNE_OUTPUT_DIR)
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def _position_bubble(self):
        """把气泡定位到桌宠上方居中。"""
        x = self.x() + self.width() // 2 - self.bubble.width() // 2
        y = self.y() - self.bubble.height() - 4
        # 防止气泡超出屏幕顶部
        screen = QApplication.primaryScreen().availableGeometry()
        if y < screen.y() + 2:
            y = screen.y() + 2
        self.bubble.move(x, y)

    def reload_knowledge(self):
        """重新加载训练文本和配置（右键菜单调用）。"""
        self.ai_manager.reload()
        n_qa = len(self.ai_manager.qa_pairs)
        n_know = len(self.ai_manager.knowledge)
        self.bubble.show_text(f"已重新加载训练文本：{n_qa}条问答，{n_know}条知识", 4000)
        self._position_bubble()

    def open_knowledge_folder(self):
        """打开训练文本文件夹（右键菜单调用）。"""
        kdir = os.path.join(app_dir(), KNOWLEDGE_DIR)
        if not os.path.exists(kdir):
            os.makedirs(kdir, exist_ok=True)
        os.startfile(kdir)  # Windows 资源管理器打开

    # ---------- 鼠标拖拽移动桌宠 ----------

    def mousePressEvent(self, event):
        self._wake()
        if event.button() == Qt.MouseButton.LeftButton:
            # 记录“鼠标全局坐标 - 窗口左上角”，移动时用它抵消，保证不偏移
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        self._wake()
        if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """双击桌宠弹出对话输入框。"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._wake()
            self.show_chat_input()
            event.accept()

    def moveEvent(self, event):
        """拖拽移动时，气泡跟随桌宠。"""
        super().moveEvent(event)
        if self.bubble.isVisible():
            self._position_bubble()

    # ---------- 右键菜单：对话 / 调整大小 / 训练文本 / 退出 ----------

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        # 子菜单“调整大小”：等比缩放档位，当前档位打勾
        size_menu = menu.addMenu("调整大小")
        for value in SCALE_OPTIONS:
            action = QAction(f"{round(value * 100)}%", size_menu)
            action.setCheckable(True)
            action.setChecked(abs(value - self.scale) < 1e-9)
            action.triggered.connect(lambda _checked, v=value: self.set_scale(v))
            size_menu.addAction(action)

        menu.addSeparator()

        # 对话
        chat_action = QAction("对话…", menu)
        chat_action.triggered.connect(self.show_chat_input)
        menu.addAction(chat_action)

        menu.addSeparator()

        # 训练文本管理
        reload_action = QAction("重新加载训练文本", menu)
        reload_action.triggered.connect(self.reload_knowledge)
        menu.addAction(reload_action)

        open_folder_action = QAction("打开训练文本文件夹", menu)
        open_folder_action.triggered.connect(self.open_knowledge_folder)
        menu.addAction(open_folder_action)

        # 显示当前AI模式
        mode = "在线AI" if self.ai_manager.api_enabled else "本地规则引擎"
        mode_action = QAction(f"AI模式：{mode}", menu)
        mode_action.setEnabled(False)
        menu.addAction(mode_action)

        menu.addSeparator()

        # 屏幕感知开关
        if self._ocr_ok:
            screen_action = QAction(f"屏幕感知（{self._ocr_engine}）", menu)
            screen_action.setCheckable(True)
            screen_action.setChecked(self._screen_perception)
            screen_action.triggered.connect(self._toggle_screen_perception)
        else:
            screen_action = QAction("屏幕感知（未安装OCR）", menu)
            screen_action.setEnabled(False)
        menu.addAction(screen_action)

        menu.addSeparator()

        # LLM 开源模型 + QLoRA 微调子菜单
        llm_menu = menu.addMenu("LLM模型")
        load_model_action = QAction("加载模型…", llm_menu)
        load_model_action.triggered.connect(self.load_llm_model)
        llm_menu.addAction(load_model_action)

        unload_model_action = QAction("卸载模型", llm_menu)
        unload_model_action.triggered.connect(self.unload_llm_model)
        unload_model_action.setEnabled(self.llm.is_loaded)
        llm_menu.addAction(unload_model_action)

        llm_menu.addSeparator()

        upload_ds_action = QAction("上传数据集…", llm_menu)
        upload_ds_action.triggered.connect(self.upload_dataset)
        llm_menu.addAction(upload_ds_action)

        open_ds_action = QAction("打开数据集文件夹", llm_menu)
        open_ds_action.triggered.connect(self.open_datasets_dir)
        llm_menu.addAction(open_ds_action)

        llm_menu.addSeparator()

        finetune_action = QAction("开始 QLoRA 微调…", llm_menu)
        finetune_action.triggered.connect(self.start_finetune)
        llm_menu.addAction(finetune_action)

        open_ft_action = QAction("打开微调输出文件夹", llm_menu)
        open_ft_action.triggered.connect(self.open_finetune_dir)
        llm_menu.addAction(open_ft_action)

        llm_menu.addSeparator()

        load_adapter_action = QAction("加载微调适配器…", llm_menu)
        load_adapter_action.triggered.connect(self.load_adapter)
        load_adapter_action.setEnabled(self.llm.is_loaded)
        llm_menu.addAction(load_adapter_action)

        unload_adapter_action = QAction("卸载微调适配器", llm_menu)
        unload_adapter_action.triggered.connect(self.unload_adapter)
        unload_adapter_action.setEnabled(bool(self.llm.adapter_path))
        llm_menu.addAction(unload_adapter_action)

        # 显示当前 LLM 状态
        if self.llm.is_loaded:
            llm_status = f"LLM：{os.path.basename(self.llm.model_path)}"
            if self.llm.adapter_path:
                llm_status += f" + {os.path.basename(self.llm.adapter_path)}"
        else:
            llm_status = "LLM：未加载"
        llm_status_action = QAction(llm_status, llm_menu)
        llm_status_action.setEnabled(False)
        llm_menu.addAction(llm_status_action)

        menu.addSeparator()

        quit_action = QAction("退出", menu)
        menu.addAction(quit_action)

        chosen = menu.exec(event.globalPos())
        if chosen is quit_action:
            QApplication.quit()


# ==================== 入口 ====================

def main():
    app = QApplication(sys.argv)

    try:
        window = PetWindow()
    except FileNotFoundError as exc:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "素材缺失", str(exc) + "\n请确认 img 文件夹与程序同级。")
        return 1

    window.show()

    # 把桌宠初始放在主屏中央偏上
    screen = app.primaryScreen().availableGeometry()
    window.move(
        screen.x() + (screen.width() - window.width()) // 2,
        screen.y() + (screen.height() - window.height()) // 3,
    )

    # 自检模式：--smoke-test 时 3 秒后自动退出（用于打包后验证 exe 能正常启动）
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(3000, app.quit)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
