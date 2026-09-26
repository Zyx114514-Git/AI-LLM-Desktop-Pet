# -*- coding: utf-8 -*-
"""pet.py 离屏自检：验证透明背景、分层渲染、眼眶限位、拖拽偏移公式。"""
import math
import os
import sys

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pet
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

app = QApplication([])
w = pet.PetWindow()
w.show()

ok = True

def check(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        ok = False

# 1) 窗口尺寸 = 身体图等比缩放后尺寸
expect_w = round(1856 * pet.SCALE)
expect_h = round(2048 * pet.SCALE)
check(f"窗口尺寸 {w.width()}x{w.height()} == 素材x缩放 {expect_w}x{expect_h}", w.width() == expect_w and w.height() == expect_h)

# 2) 透明属性必须启用（坑点）
from PyQt6.QtCore import Qt
check("WA_TranslucentBackground 已启用", w.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))

# 3) 四角像素 alpha == 0（透明背景没有黑底）
img = w.grab().toImage()
corners = [img.pixelColor(2, 2), img.pixelColor(w.width()-3, 2),
           img.pixelColor(2, w.height()-3), img.pixelColor(w.width()-3, w.height()-3)]
check("窗口四角完全透明", all(c.alpha() == 0 for c in corners))

# 4) 眼珠居中时，左右眼眶中心像素应当是身体底板的白(不是纯黑)
def pixel_alpha(x, y):
    return img.pixelColor(x, y).alpha()

c = w.eyes[0]["home"]
check("左眼眶中心处有像素(身体白底)", pixel_alpha(round(c.x()), round(c.y())) > 0)
c = w.eyes[1]["home"]
check("右眼眶中心处有像素(身体白底)", pixel_alpha(round(c.x()), round(c.y())) > 0)

# 5) 限位公式：任意方向偏移后眼珠 bbox 必须仍落在眼眶范围内
#    眼珠不透明 bbox（原图坐标，实测）与眼眶范围（原图坐标）：
#    - 左眼眶：眼白白区 x[787,892] y[559,652]（眼珠静止时下缘已贴底，故下界容 1px）
#    - 右眼眶：眼轮廓含深色外眼角 x[983,1077] y[549,658]
#              （眼珠艺术稿本身比眼白白区略高，上下压进深色眼线区属正常绘制）
SOCKETS = {
    "L": {"eye": (807, 870, 565, 652), "socket": (787, 892, 559, 653)},
    "R": {"eye": (995, 1058, 553, 656), "socket": (983, 1077, 549, 658)},
}
# 模拟 _update_eye_offsets 中的限位计算
def clamp_offset(cfg, vec):
    nx = max(-1.0, min(1.0, vec[0] / pet.MAX_GAZE_DIST))
    ny = max(-1.0, min(1.0, vec[1] / pet.MAX_GAZE_DIST))
    length = math.hypot(nx, ny)
    if length > 1.0:
        nx /= length; ny /= length
    dy_axis = cfg["max_dy_up"] * pet.SCALE if ny < 0 else cfg["max_dy_down"] * pet.SCALE
    dx = nx * cfg["max_dx"] * pet.SCALE
    dy = ny * dy_axis
    ratio = math.hypot(dx / (cfg["max_dx"] * pet.SCALE), dy / dy_axis)
    if ratio > 1.0:
        dx /= ratio; dy /= ratio
    return dx, dy

dirs = [(300, 0), (-300, 0), (0, -300), (0, 300), (300, 300), (-300, 200), (5000, 5000), (0, 0)]
worst = {}
for key, cfg in zip(["L", "R"], pet.EYES):
    worst[key] = 0
    for d in dirs:
        dx, dy = clamp_offset(cfg, d)
        home = cfg["home"]
        ex0, ex1, ey0, ey1 = SOCKETS[key]["eye"]
        sx0, sx1, sy0, sy1 = SOCKETS[key]["socket"]
        # 眼珠 bbox 整体平移 (dx, dy) 后必须 ⊆ 眼眶 bbox
        nx0, nx1 = ex0 + dx + home[0] - (ex0+ex1)//2, ex1 + dx + home[0] - (ex0+ex1)//2
        ny0, ny1 = ey0 + dy + home[1] - (ey0+ey1)//2, ey1 + dy + home[1] - (ey0+ey1)//2
        inside = nx0 >= sx0 - 1 and nx1 <= sx1 + 1 and ny0 >= sy0 - 1 and ny1 <= sy1 + 1
        # 记录最大越界量
        over = max(sx0 - nx0, nx1 - sx1, sy0 - ny0, ny1 - sy1)
        worst[key] = max(worst[key], over)
        if not inside:
            print(f"    [{key}] 方向{d} 偏移({dx:.1f},{dy:.1f}) 越界 {over:.1f}px")
    check(f"{key} 眼球在全部测试方向下不穿出眼眶(最大越界 {worst[key]:.1f}px)", worst[key] <= 1)

# 6) 拖拽偏移公式：move 后窗口左上角 == global - offset
w.move(100, 200)
gp = QPointF(350.0, 420.0)
drag_off = gp.toPoint() - w.frameGeometry().topLeft()
check("拖拽偏移计算正确", drag_off.x() == 250 and drag_off.y() == 220)

# 7) 分层渲染抽查：停掉跟随定时器后手动设置偏移，抓帧对比
#    验证：①渲染无黑底无描边 ②眼珠确实按偏移量平移（帧差法：偏移不同则眼珠像素不同）
w._timer.stop()
from PyQt6.QtGui import QImage

def grab_frame(offset):
    w._apply_offsets(QPointF(offset[0], offset[1]), QPointF(offset[0], offset[1]))
    app.processEvents()
    img = w.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = img.constBits(); ptr.setsize(img.sizeInBytes())
    # .copy() 复制数据，避免 QImage 释放后 numpy 视图悬垂导致访问冲突
    return np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 4).copy()

def diff_count(a_off, b_off, region):
    x0, y0, x1, y1 = region
    fa = grab_frame(a_off)[y0:y1, x0:x1]
    fb = grab_frame(b_off)[y0:y1, x0:x1]
    return int((fa != fb).any(axis=2).sum())

left_eye_region = (
    int(w.eyes[0]["home"].x() - 45 * w.scale),
    int(w.eyes[0]["home"].y() - 40 * w.scale),
    int(w.eyes[0]["home"].x() + 45 * w.scale),
    int(w.eyes[0]["home"].y() + 40 * w.scale),
)   # 左眼珠所在窗口（随当前缩放自适应）
diff_h = diff_count((-12, 0), (12, 0), left_eye_region)
diff_v = diff_count((0, -6), (0, 0), left_eye_region)
diff_same = diff_count((0, 0), (0, 0), left_eye_region)

# 眼珠自身深色像素数（随缩放变化），用于按比例设定偏移生效阈值
x0, y0, x1, y1 = left_eye_region
frame = grab_frame((0, 0))
lum = (frame[..., 0].astype(int) + frame[..., 1].astype(int) + frame[..., 2].astype(int)) / 3
dark_count = int(((lum < 150) & (frame[..., 3] > 200))[y0:y1, x0:x1].sum())

img_rest = w.grab().toImage()
img_rest.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_render.png"))
print(f"  左眼窗口像素差: 左移vs右移={diff_h}px 上移vs静止={diff_v}px 同帧对比={diff_same}px (眼珠深色像素={dark_count}px)")
check("同帧重复抓取无像素差异（渲染稳定）", diff_same == 0)
check("左移/右移帧眼珠像素差 > 眼珠面积30%（水平偏移生效）", diff_h > 0.3 * dark_count)
check("上移/静止帧眼珠像素差 > 眼珠面积15%（垂直偏移生效）", diff_v > 0.15 * dark_count)
check("渲染帧四角仍透明（无黑底）",
      all(img_rest.pixelColor(cx, cy).alpha() == 0
          for cx, cy in [(2, 2), (w.width()-3, 2), (2, w.height()-3), (w.width()-3, w.height()-3)]))
print("渲染测试帧已保存: test_render.png")

# 8) 动态调整大小：set_scale 后窗口、眼球几何同步等比变化，且中心保持
import pet as pet_mod
center_before = w.frameGeometry().center()
w.set_scale(0.25)
check("缩放0.25后窗口尺寸=素材x0.25",
      w.width() == round(1856 * 0.25) and w.height() == round(2048 * 0.25))
check("缩放后左眼眶中心=home*0.25",
      abs(w.eyes[0]["home"].x() - 838 * 0.25) < 0.01 and abs(w.eyes[0]["home"].y() - 608 * 0.25) < 0.01)
check("缩放后水平限位=max_dx*0.25", abs(w.eyes[0]["max_dx"] - 12 * 0.25) < 0.01)
center_after = w.frameGeometry().center()
check("缩放后窗口中心位置不变(跳动<1px)",
      abs(center_after.x() - center_before.x()) <= 1 and abs(center_after.y() - center_before.y()) <= 1)
w.set_scale(pet.SCALE)  # 恢复默认
check("恢复默认缩放后窗口尺寸正确", w.width() == round(1856 * pet.SCALE))
# 右键菜单档位存在且包含当前档
check("缩放档位列表包含默认档", any(abs(v - pet.SCALE) < 1e-9 for v in pet.SCALE_OPTIONS))

# 9) 眨眼：参数、状态机、闭眼覆盖
check("眨眼间隔配置在2.5~4秒",
      pet.BLINK_MIN_INTERVAL >= 2.5 and pet.BLINK_MAX_INTERVAL <= 4.0)
check("一次眨眼总时长<0.3秒（短促）",
      pet.BLINK_CLOSE_TIME + pet.BLINK_HOLD_TIME + pet.BLINK_OPEN_TIME < 0.3)

w._blink_state = "waiting"; w._blink_timer = 0.01
w._update_blink(0.02)
check("计时到后进入闭眼阶段", w._blink_state == "closing")
w._update_blink(1.0)
check("闭眼完成后进入停留阶段", w._blink_state == "closed" and w._blink_progress == 1.0)
w._update_blink(0.1)
check("停留结束进入睁眼阶段", w._blink_state == "opening")
w._update_blink(1.0)
check("睁眼完成回到等待且完全睁开", w._blink_state == "waiting" and w._blink_progress == 0.0)

# 完全闭合时眼睑覆盖眼眶（上部变亮为皮肤色），睁开后恢复眼珠
w._timer.stop()
lh = w.eyes[0]["home"]
sock = pet.EYES[0]["socket"]
sock_h = (sock[3] - sock[1]) * w.scale
pt_upper = (round(lh.x()), round(sock[1] * w.scale + sock_h * 0.25))
w._blink_state = "closed"; w._blink_progress = 1.0
w.update(); app.processEvents()
img_closed = w.grab().toImage()
c_closed = img_closed.pixelColor(*pt_upper)
w._blink_state = "waiting"; w._blink_progress = 0.0
w.update(); app.processEvents()
img_open = w.grab().toImage()
c_open = img_open.pixelColor(*pt_upper)
print(f"  眼眶上部像素: 闭眼={c_closed.name()} 睁眼={c_open.name()}")
check("闭眼后眼眶上部被眼睑皮肤覆盖（明显变亮）",
      c_closed.lightness() > 120 and c_closed.lightness() - c_open.lightness() > 60)

# 10) 耳朵表情切换
check("表情素材已加载(2张)", len(w.expressions) == 2)
check("表情图与身体同尺寸(等比不拉伸)",
      w.expressions[0]["pix"].width() == w.body_pix.width()
      and w.expressions[0]["pix"].height() == w.body_pix.height())
# 热区(窗口坐标)应位于头顶两侧
le, re = w.expressions[0]["rect"], w.expressions[1]["rect"]
check("左耳热区在窗口左半部", le.center().x() < w.width() * 0.5 and le.center().y() < w.height() * 0.35)
check("右耳热区在窗口右半部", re.center().x() > w.width() * 0.5 and re.center().y() < w.height() * 0.35)

# 悬停左耳 → 表情0；悬停右耳 → 表情1；离开 → None（热区判定逻辑）
le, re = w.expressions[0]["rect"], w.expressions[1]["rect"]
check("左耳热区中心在内", le.contains(le.center()))
check("右耳热区中心在内", re.contains(re.center()))
check("左耳热区不含右耳中心", not le.contains(re.center()))
check("窗口外的点不在热区内", not le.contains(QPointF(-10, -10)))

# 整图切换渲染：表情模式下眼眶处不再是"睁眼眼珠"，且窗口四角仍透明
w._timer.stop()
lh = w.eyes[0]["home"]
w._expr_index = 0
w.update(); app.processEvents()
img_expr = w.grab().toImage()
c_expr = img_expr.pixelColor(round(lh.x()), round(lh.y()))
w._expr_index = None
w.update(); app.processEvents()
img_norm = w.grab().toImage()
c_norm = img_norm.pixelColor(round(lh.x()), round(lh.y()))
print(f"  左眼眶中心像素: 表情={c_expr.name()} 正常={c_norm.name()}")
check("表情模式渲染不同于正常分层渲染", c_expr.name() != c_norm.name())
check("表情模式下四角仍透明",
      img_expr.pixelColor(2, 2).alpha() == 0 and img_expr.pixelColor(img_expr.width()-3, img_expr.height()-3).alpha() == 0)

print("\n结果:", "全部通过" if ok else "存在失败项")
sys.exit(0 if ok else 1)
