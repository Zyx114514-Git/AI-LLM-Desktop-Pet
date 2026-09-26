# -*- coding: utf-8 -*-
"""
垂耳表情素材预处理（一次性工具）：
在闭眼微笑表情图（expr_left/expr_right，已由 prepare_expressions.py 生成）基础上，
擦除原立耳、叠加 ear_fall.png 垂耳（垂耳根部对齐原耳朵根部，向下耷拉），生成最终表情图：
  expr_left_final.png  —— 左耳悬停：闭眼微笑 + 左垂耳
  expr_right_final.png —— 右耳悬停：闭眼微笑 + 右垂耳

处理流程：
  1. 加载 ear_fall.png（右垂耳），水平翻转得到左垂耳；
  2. ear_fall 耳朵根部在左上角，向右下耷拉；把根部对齐到原耳朵根部（底部中心），
     垂耳自然从耳朵位置向下耷拉、贴在头部两侧；
  3. 加载表情图，在耳朵区域内擦除非白色像素（保留白色发饰），阈值 235；
  4. 叠加对应垂耳；
  5. 输出最终表情图。

用法：python prepare_ear_fall.py
（需先运行 prepare_expressions.py 生成 expr_left.png / expr_right.png）
"""
import os

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
EAR_FALL_SRC = r"d:\Users\Administrator\Downloads\ear_fall.png"
WHITE_TH = 235  # 低于此值视为非白色（耳朵/描边），擦除；高于此值保留（白色发饰）

# 耳朵擦除区域（原图坐标，略大于耳朵范围，确保描边完全擦除）
EAR_REGIONS = {
    "left":  (640, 200, 800, 425),   # 左耳
    "right": (1045, 195, 1200, 425),  # 右耳
}

# 垂耳偏移量（dx, dy）：把 ear_fall 耳朵从原始位置移动到目标位置
# ear_fall 右耳原始位置 x[1139,1280] y[353,536]，根部在左上(1139,353)
# 对齐到原右耳根部(1123,409)，偏移 (-16, 56)；再左移20px更贴合 => (-36, 56)
# 左耳翻转后原始位置 x[576,717] y[353,536]，根部在右上(717,353)
# 对齐到原左耳根部(717,409)，偏移 (0, 56)；再右移20px更贴合 => (20, 56)
EAR_OFFSETS = {
    "left":  (20, 56),
    "right": (-36, 56),
}


def erase_ear(img: Image.Image, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    """在指定区域内擦除非白色像素（保留白色发饰），返回新图。"""
    a = np.array(img)
    rgb = a[..., :3].astype(int)
    region = a[y0:y1, x0:x1]
    r_rgb = rgb[y0:y1, x0:x1]
    nonwhite = (r_rgb[..., 0] < WHITE_TH) | (r_rgb[..., 1] < WHITE_TH) | (r_rgb[..., 2] < WHITE_TH)
    region[nonwhite, 3] = 0
    a[y0:y1, x0:x1] = region
    return Image.fromarray(a, "RGBA")


def move_ear(ear_img: Image.Image, dx: int, dy: int) -> Image.Image:
    """把耳朵素材平移 (dx, dy)，返回同尺寸新图。"""
    new = Image.new("RGBA", ear_img.size, (0, 0, 0, 0))
    new.paste(ear_img, (dx, dy), ear_img)
    return new


def main():
    ear_fall_right = Image.open(EAR_FALL_SRC).convert("RGBA")
    ear_fall_left = ear_fall_right.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    for side, expr_name, out_name, ear_pix in [
        ("left",  "expr_left.png",  "expr_left_final.png",  ear_fall_left),
        ("right", "expr_right.png", "expr_right_final.png", ear_fall_right),
    ]:
        expr_path = os.path.join(IMG_DIR, expr_name)
        if not os.path.exists(expr_path):
            print(f"跳过（缺少 {expr_name}，请先运行 prepare_expressions.py）")
            continue
        expr = Image.open(expr_path).convert("RGBA")
        x0, y0, x1, y1 = EAR_REGIONS[side]
        expr = erase_ear(expr, x0, y0, x1, y1)
        dx, dy = EAR_OFFSETS[side]
        expr.alpha_composite(move_ear(ear_pix, dx, dy))
        out_path = os.path.join(IMG_DIR, out_name)
        expr.save(out_path)
        print(f"已生成 {out_name}（{side}耳：擦除立耳 + 垂耳根部对齐向下耷拉，偏移({dx},{dy})）")


if __name__ == "__main__":
    main()
