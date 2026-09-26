# -*- coding: utf-8 -*-
"""
屏幕感知工具：截图 + OCR 文字识别
==================================

功能：
  1. 截取当前全屏（或主显示器）
  2. 对截图做 OCR 提取文字（优先 Windows 自带 OCR (winocr)，备选 pytesseract）
  3. 识别结果作为对话上下文，让 AI "看到"用户屏幕上的内容

依赖（可选）：
  pip install winocr          # Windows 10+ 自带 OCR，推荐（无需额外装 Tesseract）
  pip install pytesseract      # 备选，需另外安装 Tesseract-OCR 程序
  两个都没装时，屏幕感知功能不可用，程序仍可正常运行。
"""

import io
import os
import sys
import tempfile


def capture_screen():
    """
    截取主显示器全屏，返回 PIL.Image.Image（RGB 格式）。
    使用 PyQt6 的 QScreen.grabWindow(0)，无需额外依赖。
    """
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QRect

    screen = QApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("无法获取主屏幕")
    pixmap = screen.grabWindow(0)
    # QPixmap -> PIL Image
    qimg = pixmap.toImage()
    # 确保格式为 RGB888
    qimg = qimg.convertToFormat(qimg.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    bits = qimg.bits()
    bits.setsize(h * w * 3)
    from PIL import Image
    img = Image.frombytes("RGB", (w, h), bits, "raw", "RGB", 0, 1)
    return img


def check_ocr_available() -> tuple:
    """检测 OCR 是否可用，返回 (是否可用, 引擎名称或缺失提示)。"""
    try:
        import winocr  # noqa: F401
        return (True, "winocr (Windows自带OCR)")
    except ImportError:
        pass
    try:
        import pytesseract  # noqa: F401
        # 检查 tesseract 是否实际可用
        try:
            pytesseract.get_tesseract_version()
            return (True, "pytesseract (Tesseract)")
        except Exception:
            return (False, "pytesseract 已安装但 Tesseract-OCR 程序未找到")
    except ImportError:
        pass
    return (False, "未安装 OCR 引擎（pip install winocr 或 pytesseract）")


def ocr_image(pil_image) -> str:
    """
    对 PIL 图片做 OCR，返回识别到的文字（空字符串表示无文字或失败）。
    优先 winocr（Windows 自带），备选 pytesseract。
    """
    # 方案1: winocr（Windows 10+ 自带 OCR）
    try:
        import asyncio
        import winocr
        from PIL import Image

        # winocr 需要保存为临时 PNG 文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            pil_image.save(tmp_path, format="PNG")
            # winocr 是异步 API
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(winocr.recognize_pil(pil_image))
            finally:
                loop.close()
            text = result.text or ""
            return text.strip()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except ImportError:
        pass
    except Exception:
        pass

    # 方案2: pytesseract
    try:
        import pytesseract
        text = pytesseract.image_to_string(pil_image, lang="chi_sim+eng")
        return text.strip()
    except Exception:
        pass

    return ""


def get_screen_context(max_chars: int = 1500) -> str:
    """
    截取屏幕并 OCR，返回屏幕文字摘要（供 AI 作为上下文）。
    失败时返回空字符串。
    """
    try:
        img = capture_screen()
    except Exception:
        return ""

    try:
        text = ocr_image(img)
    except Exception:
        return ""

    if not text:
        return ""

    # 压缩空白行，限制长度
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


if __name__ == "__main__":
    # 测试：直接运行此文件查看截图 OCR 效果
    print("截取屏幕中...")
    img = capture_screen()
    print(f"截图尺寸: {img.size}")
    ok, engine = check_ocr_available()
    print(f"OCR 可用: {ok} ({engine})")
    if ok:
        print("OCR 识别中...")
        text = ocr_image(img)
        print("--- 识别结果 ---")
        print(text[:1000])
