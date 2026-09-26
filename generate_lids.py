# -*- coding: utf-8 -*-
"""
生成闭眼眼睑素材（一次性工具）：从身体底板的眼周颜色与眼眶几何程序化生成
img/eye_lid_left.png、img/eye_lid_right.png（全画布 1856x2048 透明 PNG，
与其它素材同坐标系，直接叠放即对齐）。

闭眼结构（每个眼眶内）：
  上部 = 上眼睑皮肤色（采样自素材）
  中部 = 深色睫毛线（贴近素材睫毛色，外眼角端微微下垂）
  下部 = 下眼睑皮肤色（采样自素材）
形状 = 覆盖眼眶开口的水平椭圆（比眼白白区略大 3px，完全闭合时不留白边）。
"""
import os
import numpy as np
from PIL import Image

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
BODY = os.path.join(IMG_DIR, "idle_empty.png")

# 各眼几何与配色（数值均来自对素材的实测采样）
# socket: (x0, y0, x1, y1) 眼眶白区范围；outer_side: 外眼角所在侧
EYE_SPECS = [
    dict(name="left", out="eye_lid_left.png", socket=(787, 559, 892, 652),
         upper=(248, 216, 169), lower=(253, 250, 219), lash=(20, 10, 8), outer_side="left"),
    dict(name="right", out="eye_lid_right.png", socket=(986, 555, 1065, 652),
         upper=(253, 252, 221), lower=(253, 251, 219), lash=(23, 11, 8), outer_side="right"),
]

CANVAS = (1856, 2048)
LASH_THICK = 6          # 睫毛线厚度(px)
LASH_DIP = 5            # 外眼角端下垂幅度(px)
MARGIN = 3              # 比眼眶白区外扩的像素，确保完全闭合


def build_lid(spec):
    w, h = CANVAS
    x0, y0, x1, y1 = spec["socket"]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = (x1 - x0) / 2.0 + MARGIN
    ry = (y1 - y0) / 2.0 + MARGIN

    ys, xs = np.mgrid[0:h, 0:w]
    # 1) 眼眶形状：水平椭圆掩码
    ellipse = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0

    # 2) 睫毛线位置：随 x 从内眼角到外眼角线性下垂（贴近素材眼线走向）
    t = (xs - x0) / (x1 - x0) if spec["outer_side"] == "right" else (x1 - xs) / (x1 - x0)
    lash_y = cy + LASH_DIP * t   # t=0 内眼角(高)，t=1 外眼角(低)
    lash_band = (ys >= lash_y - LASH_THICK / 2.0) & (ys <= lash_y + LASH_THICK / 2.0)

    # 3) 填色：上=上睑皮肤，中=睫毛线，下=下睑皮肤；透明处 alpha=0
    lid = np.zeros((h, w, 4), dtype=np.uint8)
    for yy in range(h):
        row = ellipse[yy]
        if not row.any():
            continue
        lash_row = lash_y[yy]  # 该行的睫毛线 y（一维）
        band = lash_band[yy] & row
        up = row & (ys[yy] < lash_row - LASH_THICK / 2.0)
        low = row & (ys[yy] > lash_row + LASH_THICK / 2.0)
        lid[yy, band, :3] = spec["lash"]
        lid[yy, up, :3] = spec["upper"]
        lid[yy, low, :3] = spec["lower"]
        lid[yy, row, 3] = 255

    # 4) 边缘羽化：仅对 alpha 通道做高斯模糊（颜色不变），让眼睑与画面自然融合
    from PIL import ImageFilter
    alpha = Image.fromarray(lid[..., 3], "L")
    alpha = alpha.filter(ImageFilter.GaussianBlur(1.5))
    lid[..., 3] = np.array(alpha)

    return Image.fromarray(lid, "RGBA")


def main():
    for spec in EYE_SPECS:
        img = build_lid(spec)
        out = os.path.join(IMG_DIR, spec["out"])
        img.save(out)
        print(f"已生成 {spec['out']} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
