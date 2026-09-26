# -*- coding: utf-8 -*-
"""
表情素材预处理（一次性工具）：把用户上传的“闭眼微笑”全身图（白底 RGB）转换为
透明 PNG（全画布 1856x2048，与其它素材同坐标系）。

难点：上传图是白底 RGB，角色本身又含大量纯白衣物（围裙/蕾丝），直接“去白底”
会把衣物打成透明洞；因此以身体底板 idle_empty.png 的 alpha 作为角色轮廓种子，
再沿上传图的彩色内容做区域生长，得到完整的角色蒙版：

  1. 种子 = idle_empty 的 alpha（同一角色、同画布同位置，轮廓可靠）；
  2. 生长 = 种子沿上传图“非白像素”四连通扩张——吸收上传图超出底板轮廓的部分
     （更大的头发/裙摆等），而右下角水印与角色不连通，自动被排除；
  3. 蒙版内像素：上传图有明确彩色内容处用上传图颜色（表情细节），
     白色衣物/轮廓边缘处用底板的颜色与 alpha（保证不透明、无白边、边缘柔和）；
  4. 输出 img/expr_left.png、img/expr_right.png。

用法：python prepare_expressions.py
"""
import os

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
IDLE_PATH = os.path.join(IMG_DIR, "idle_empty.png")
STRONG_TH = 245  # 上传图 min_ch < 此值视为“有明确彩色内容”

SOURCES = [
    (r"d:\Users\Administrator\Downloads\变成人形 (5).png", "expr_left.png"),   # 左耳触发
    (r"d:\Users\Administrator\Downloads\变成人形 (5).png", "expr_right.png"),  # 右耳触发（当前统一用同一张图）
]


def _grow(seed: np.ndarray, strong: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """seed 沿 strong 像素四连通扩张（迭代形态学膨胀），返回扩张后的蒙版。"""
    mask = seed.copy()
    for _ in range(max_iter):
        d = np.zeros_like(mask)
        d[1:, :] |= mask[:-1, :]
        d[:-1, :] |= mask[1:, :]
        d[:, 1:] |= mask[:, :-1]
        d[:, :-1] |= mask[:, 1:]
        new = d & strong & ~mask
        if not new.any():
            break
        mask |= new
    return mask


def make_transparent(src_path: str) -> Image.Image:
    upload = Image.open(src_path).convert("RGB")
    idle = Image.open(IDLE_PATH).convert("RGBA")
    assert upload.size == idle.size, f"尺寸不一致: {upload.size} vs {idle.size}"

    up = np.array(upload).astype(int)
    id_rgb = np.array(idle)[..., :3].astype(int)
    id_alpha = np.array(idle)[..., 3].astype(int)

    min_ch = np.minimum(np.minimum(up[..., 0], up[..., 1]), up[..., 2])
    strong = min_ch < STRONG_TH  # 上传图明确内容（非白、非近白）

    seed = id_alpha > 10         # 底板轮廓种子
    mask = _grow(seed, strong)   # 生长吸收上传图超出底板的部分

    # RGB：有明确内容处用上传图（表情细节），否则用底板（白色衣物/边缘颜色）
    rgb = np.where(strong[..., None], up, id_rgb)
    # alpha：上传图内容处 255；底板覆盖处沿用底板 alpha（白色衣物不透明、轮廓边缘柔和）
    alpha = np.where(mask & strong, 255, 0).astype(np.uint8)
    alpha = np.where(mask & ~strong, id_alpha.astype(np.uint8), alpha)

    out = np.dstack([rgb.astype(np.uint8), alpha]).astype(np.uint8)
    print(f"  蒙版: 种子{int(seed.sum())}px → 生长后{int(mask.sum())}px"
          f"（+{int(mask.sum() - seed.sum())}px）；水印区(y>1900)残留 {int((mask[1900:] & ~(seed[1900:])).sum())}px")
    return Image.fromarray(out, "RGBA")


def main():
    for src, out_name in SOURCES:
        if not os.path.exists(src):
            print(f"跳过（不存在）: {src}")
            continue
        img = make_transparent(src)
        out = os.path.join(IMG_DIR, out_name)
        img.save(out)
        print(f"已生成 {out_name} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
