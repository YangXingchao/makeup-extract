import argparse
import os

import cv2
import numpy as np
import torch

import config
from networks import MakeupExtractNet
from utils.img_util import save_tensor
from utils.mm_layer import get_cropped_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='makeup extraction')
    parser.add_argument('--uv_albe_path', default='./results/material_refinement/albe_refine.png',
                        help='Input refined albedo')
    parser.add_argument('--out_dir', '-o', default='./results/makeup_extraction', help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    tex_size = config.UV_TEX_SIZE

    net = MakeupExtractNet(3, 7).to(device)
    net.load_state_dict(torch.load('checkpoints/makeup_extraction.pkl'))
    net.eval()

    cropped_mask = get_cropped_mask(device)

    makeup_albe = cv2.imread(args.uv_albe_path)
    makeup_albe = cv2.resize(makeup_albe, (tex_size, tex_size))
    makeup_albe = torch.Tensor(makeup_albe.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    makeup_albe = makeup_albe * 2. - 1

    make_res = net(makeup_albe)
    bare_skin = make_res[:, :3, ...].clamp(-1, 1)
    make_base = make_res[:, 3:6, ...].clamp(-1, 1)
    make_alpha = make_res[:, 6:, ...].clamp(-1, 1)

    make_alpha = make_alpha * 0.5 + 0.5
    make_blend = (make_base * 0.5 + 0.5) * (1 - make_alpha)

    bare_skin = ((bare_skin * 0.5 + 0.5) * cropped_mask).permute(0, 2, 3, 1)[0]
    make_base = ((make_base * 0.5 + 0.5) * cropped_mask).permute(0, 2, 3, 1)[0]
    make_alpha = ((1-make_alpha.expand(-1, 3, -1, -1)) * cropped_mask).permute(0, 2, 3, 1)[0]
    make_blend = (make_blend * cropped_mask).permute(0, 2, 3, 1)[0]

    save_tensor(bare_skin, f'{out_dir}/bare_skin.png')
    save_tensor(make_base, f'{out_dir}/make_base.png')
    save_tensor(make_alpha, f'{out_dir}/make_alpha.png')
    save_tensor(make_blend, f'{out_dir}/make_blend.png')
