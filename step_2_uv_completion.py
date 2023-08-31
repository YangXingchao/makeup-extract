import argparse
import os

import cv2
import numpy as np
import torch

import config
from networks import UVCompletionNet
from utils.mm_layer import get_cropped_mask
from utils.img_util import save_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='uv completion')
    parser.add_argument('--uv_mm_path', default='./results/coarse_reconstruction/uv_rncr.png',
                        help='Input unwrapped 3DMM image')
    parser.add_argument('--uv_tex_path', default='./results/coarse_reconstruction/uv_tex.png',
                        help='Input unwrapped uv texture')
    parser.add_argument('--out_dir', '-o', default='./results/material_refinement', help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    tex_size = config.UV_TEX_SIZE

    cropped_mask = get_cropped_mask(device)

    mm_tex = cv2.imread(args.uv_mm_path)
    mm_tex = cv2.resize(mm_tex, (tex_size, tex_size))
    mm_tensor = torch.tensor(mm_tex.astype(np.float32) / 255.0).permute(2, 0, 1)
    mm_tensor = mm_tensor * 2. - 1

    tex = cv2.imread(args.uv_tex_path)
    tex = cv2.resize(tex, (tex_size, tex_size))
    input_tensor = torch.tensor(tex.astype(np.float32) / 255.0).permute(2, 0, 1)
    input_tensor = input_tensor * 2. - 1

    gray_tex = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
    _, binary_tex = cv2.threshold(gray_tex, 10, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(binary_tex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
    center = tuple(
        np.mean([np.min(max_cnt, axis=0).squeeze(), np.max(max_cnt, axis=0).squeeze()], axis=0, dtype=np.int32))
    input_tex = cv2.seamlessClone(tex, mm_tex, binary_tex, center, cv2.NORMAL_CLONE)

    tex_fill = np.zeros_like(binary_tex)
    cv2.drawContours(tex_fill, [max_cnt], -1, color=255, thickness=-1)

    tex_erode = cv2.erode(tex_fill, np.ones((11, 11), np.uint8), iterations=5)
    tex_blur = cv2.GaussianBlur(tex_erode, (15, 15), 11)
    tex_erode_blur = cv2.bitwise_and(tex_blur, tex_fill)
    tex_rec = cv2.bitwise_and(binary_tex, tex_erode_blur)

    tex_rec = torch.Tensor(tex_rec.astype(np.float32) / 255.0)
    tex_rec = torch.stack([tex_rec, tex_rec, tex_rec], dim=2).permute(2, 0, 1)

    input_tensor = input_tensor * tex_rec + mm_tensor * (1 - tex_rec)

    net = UVCompletionNet().to(device)
    net.load_state_dict(torch.load('checkpoints/uv_completion.pkl'))
    net.eval()

    input_tensor = torch.cat([input_tensor, torch.flip(input_tensor, [2])], dim=0).to(device)
    res_tex = net(input_tensor[None] * cropped_mask)
    res_tex = res_tex.clamp(-1, 1) * cropped_mask
    out_tensor = res_tex[0].permute(1, 2, 0) * 0.5 + 0.5
    save_tensor(out_tensor, f'{out_dir}/completion.png')
