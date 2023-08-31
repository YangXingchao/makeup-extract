import argparse
import os

import cv2
import kornia
import numpy as np
import nvdiffrast.torch as dr
import torch

import config
from utils.img_util import save_tensor
from utils.loss_util import VGGLoss, TVLoss
from utils.mm_layer import get_mm, get_cropped_mask
from utils.mm_util import compute_mm
from utils.render_util import diffuse_shading


def load_tex(tex_path, tex_w, tex_h):
    uv_tex = cv2.imread(tex_path)
    uv_tex = cv2.resize(uv_tex, (tex_w, tex_h))
    tex_tensor = torch.Tensor(uv_tex.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tex_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='material refinement')
    parser.add_argument('--mm_param_path', default='./results/coarse_reconstruction/mm_param.npz',
                        help='Input fitted 3DMM parameter')
    parser.add_argument('--uv_albe_path', default='./results/coarse_reconstruction/uv_albe.png',
                        help='Input unwrapped 3DMM albe')
    parser.add_argument('--uv_norm_path', default='./results/coarse_reconstruction/uv_norm.png',
                        help='Input unwrapped 3DMM norm')
    parser.add_argument('--uv_rnsr_path', default='./results/coarse_reconstruction/uv_rnsr.png',
                        help='Input unwrapped 3DMM rnsr')
    parser.add_argument('--completion_tex_path', default='./results/material_refinement/completion.png',
                        help='Input unwrapped uv texture')
    parser.add_argument('--n_itr', type=int, default=500, help='Input iteration number')
    parser.add_argument('--lr', type=float, default=1e-2, help='Input learning rate')
    parser.add_argument('--out_dir', '-o', default='./results/material_refinement', help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # parameters
    n_itr = args.n_itr
    lr = args.lr

    w_dic = {'w_tex': 40., 'w_albe': 4., 'w_norm': 1., 'w_rnsr': 1., 'w_sh': 1., 'w_vgg': 5., 'w_tv': 10.}

    fit_size = config.FIT_SIZE
    tex_w = config.UV_TEX_SIZE
    tex_h = config.UV_TEX_SIZE
    uv_crop_w = config.UV_CROP_W
    uv_crop_h = config.UV_CROP_H
    uv_size = config.UNWARP_UV_SIZE * tex_w // uv_crop_w

    # get mm components
    mm = get_mm()
    shape_layer = mm['shape_layer'].to(device)
    tex_layer = mm['tex_layer'].to(device)
    spec_layer = mm['spec_layer'].to(device)
    tri = mm['tri'].to(device)
    uvs = mm['uvs'].to(device)
    uv_ids = mm['uv_ids'].to(device)
    uv_coords = mm['uv_coords'].to(device)

    uv_crop_pos = [config.UV_CROP_POS[0] * tex_w / uv_crop_w, config.UV_CROP_POS[1] * tex_h / uv_crop_h]
    uv_crop_pos = np.around(uv_crop_pos).astype(np.int32)

    glctx = dr.RasterizeGLContext()
    rast_uv, _ = dr.rasterize(glctx, uv_coords.contiguous(), uv_ids.contiguous(), resolution=[uv_size, uv_size])

    fit_uv_mask = cv2.imread(config.SKIN_MASK_PATH)
    rend_uv_mask = cv2.resize(fit_uv_mask, (uv_size, uv_size))
    rend_uv_mask = torch.tensor(rend_uv_mask.astype(np.float32) / 255.0)[None].to(device)

    cropped_mask = get_cropped_mask(device)

    err_l1 = torch.nn.L1Loss().to(device)
    err_mse = torch.nn.MSELoss().to(device)
    err_vgg = VGGLoss().to(device)
    err_tv = TVLoss().to(device)

    # load gt data
    albe_gt = load_tex(args.uv_albe_path, tex_w, tex_h).to(device)
    norm_gt = load_tex(args.uv_norm_path, tex_w, tex_h).to(device) * 2. - 1
    rnsr_gt = load_tex(args.uv_rnsr_path, tex_w, tex_h).to(device)
    tex_gt = load_tex(args.completion_tex_path, tex_w, tex_h).to(device)

    # mm param
    fit_param = np.load(args.mm_param_path)
    id = torch.from_numpy(fit_param['id']).float().to(device)
    ex = torch.from_numpy(fit_param['ex']).float().to(device)
    tx = torch.from_numpy(fit_param['tx']).float().to(device)
    sp = torch.from_numpy(fit_param['sp']).float().to(device)
    r = torch.from_numpy(fit_param['r']).float().to(device)
    tr = torch.from_numpy(fit_param['tr']).float().to(device)
    s = torch.from_numpy(fit_param['s']).float().to(device)
    sh = torch.from_numpy(fit_param['sh']).float().to(device)
    p = torch.from_numpy(fit_param['p']).float().to(device)
    ln = torch.from_numpy(fit_param['ln']).float().to(device)
    gain = torch.from_numpy(fit_param['gain']).float().to(device)
    bias = torch.from_numpy(fit_param['bias']).float().to(device)

    albe_opt = albe_gt[None].clone().detach().requires_grad_(True)
    norm_opt = norm_gt[None].clone().detach().requires_grad_(True)
    rnsr_gt = kornia.color.bgr_to_grayscale(rnsr_gt)
    rnsr_opt = rnsr_gt[None].clone().detach().requires_grad_(True)
    sh_opt = sh.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([albe_opt, norm_opt, rnsr_opt, sh_opt], lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda x: 0.1 ** (float(x) / float(n_itr)))

    # start refine iteration
    for itr in range(n_itr):
        mm_ret = compute_mm(shape_layer, tex_layer, spec_layer, tri, id, ex, tx, sp, r, tr, s, sh_opt, p, ln, gain,
                            bias)

        norm_opt = norm_opt.permute(0, 2, 3, 1).reshape(1, -1, 3)
        _, diff = diffuse_shading(norm_opt, sh_opt)
        diff = diff.permute(0, 2, 1).reshape(1, 3, tex_w, tex_h)
        norm_opt = norm_opt.permute(0, 2, 1).reshape(1, 3, tex_w, tex_h)
        rndr_opt = albe_opt * diff

        rnsr_opt = rnsr_opt.expand(-1, 3, -1, -1)

        tex_opt = rndr_opt + rnsr_opt

        blur_albe_opt = kornia.filters.gaussian_blur2d(albe_opt, (5, 5), (3, 3))
        blur_norm_opt = kornia.filters.gaussian_blur2d(norm_opt, (5, 5), (3, 3))
        blur_rnsr_opt = kornia.filters.gaussian_blur2d(rnsr_opt, (5, 5), (3, 3))

        # loss
        loss_tex = err_l1(tex_opt, tex_gt) * w_dic['w_tex']
        loss_albe = err_l1(blur_albe_opt * cropped_mask, albe_gt * cropped_mask) * w_dic['w_albe']
        loss_norm = err_l1(blur_norm_opt * cropped_mask, albe_gt * cropped_mask) * w_dic['w_norm']
        loss_rnsr = err_l1(blur_rnsr_opt * cropped_mask, rnsr_gt * cropped_mask) * w_dic['w_rnsr']
        loss_able_vgg = err_vgg(tex_opt * cropped_mask, tex_gt * cropped_mask) * w_dic['w_vgg']
        loss_sh = err_mse(sh_opt, sh) * w_dic['w_sh']
        loss_norm_tv = err_tv(rnsr_opt * cropped_mask) * w_dic['w_tv']
        loss_norm_tv += err_tv(norm_opt * cropped_mask) * w_dic['w_tv']
        loss_norm_tv += err_tv(albe_opt * cropped_mask) * w_dic['w_tv']
        loss = loss_tex + loss_albe + loss_norm + loss_rnsr + loss_able_vgg + loss_sh + loss_norm_tv

        # optimizer update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'itr [{itr:04}/{n_itr - 1:04}]', loss.item())

    # save results
    tex_refine = (tex_opt.clamp(0, 1.) * cropped_mask).permute(0, 2, 3, 1)[0]
    albe_refine = (albe_opt.clamp(0, 1.) * cropped_mask).permute(0, 2, 3, 1)[0]
    norm_refine = (norm_opt.clamp(-1, 1.) * cropped_mask).permute(0, 2, 3, 1)[0] * 0.5 + 0.5
    diff_refine = (diff.clamp(0, 1.) * cropped_mask).permute(0, 2, 3, 1)[0]
    rnsr_refine = (rnsr_opt.clamp(0, 1.) * cropped_mask).permute(0, 2, 3, 1)[0]

    save_tensor(tex_refine, f'{out_dir}/tex_refine.png')
    save_tensor(albe_refine, f'{out_dir}/albe_refine.png')
    save_tensor(norm_refine, f'{out_dir}/norm_refine.png')
    save_tensor(diff_refine, f'{out_dir}/diff_refine.png')
    save_tensor(rnsr_refine, f'{out_dir}/rnsr_refine.png')
