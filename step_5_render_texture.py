import argparse
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

import config
from utils.img_util import save_tensor
from utils.mm_layer import get_mm
from utils.mm_util import compute_mm
from utils.render_util import compute_img_render, compute_uv_render


def get_uncrop_tex(tex_path, mask_region, tex_w, tex_h, uv_size, uv_crop_pos):
    tex_raw = cv2.imread(tex_path)
    tex_raw = cv2.resize(tex_raw, (tex_w, tex_h))
    tex_raw = tex_raw * mask_region
    tex = np.zeros([uv_size, uv_size, 3])
    tex[uv_crop_pos[1]:uv_crop_pos[1] + tex_h, uv_crop_pos[0]:uv_crop_pos[0] + tex_w, :] = tex_raw
    tex = torch.Tensor(tex.astype(np.float32) / 255.0)
    return tex


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='render texture')
    parser.add_argument('--mm_param_path', default='./results/coarse_reconstruction/mm_param.npz',
                        help='Input fitted 3DMM parameter')
    parser.add_argument('--uv_bare_skin_path', default='./results/makeup_extraction/bare_skin.png',
                        help='Input bare skin')
    parser.add_argument('--uv_make_alpha_path', default='./results/makeup_extraction/make_alpha.png',
                        help='Input alpha matte')
    parser.add_argument('--uv_make_blend_path', default='./results/makeup_extraction/make_blend.png',
                        help='Input makeup blend')
    parser.add_argument('--uv_diffuse_shading_path', default='./results/material_refinement/diff_refine.png',
                        help='Input diffuse shading')
    parser.add_argument('--uv_specular_shading_path', default='./results/material_refinement/rnsr_refine.png',
                        help='Input diffuse shading')
    parser.add_argument('--out_dir', '-o', default='./results/render_texture', help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    fit_size = config.FIT_SIZE
    unwarp_uv_size = config.UNWARP_UV_SIZE
    render_size = config.RENDER_SIZE
    uv_crop_pos = config.UV_CROP_POS
    uv_crop_w = config.UV_CROP_W
    uv_crop_h = config.UV_CROP_H

    # get mm components
    mm = get_mm()
    shape_layer = mm['shape_layer'].to(device)
    tex_layer = mm['tex_layer'].to(device)
    spec_layer = mm['spec_layer'].to(device)
    tri = mm['tri'].to(device)
    uvs = mm['uvs'].to(device)
    uv_coords = mm['uv_coords'].to(device)
    uv_ids = mm['uv_ids'].to(device)

    glctx = dr.RasterizeGLContext()
    rast_uv, _ = dr.rasterize(glctx, uv_coords.contiguous(), uv_ids.contiguous(), resolution=[unwarp_uv_size, unwarp_uv_size])

    fit_uv_mask = cv2.imread(config.SKIN_MASK_PATH)
    rend_uv_mask = cv2.resize(fit_uv_mask, (unwarp_uv_size, unwarp_uv_size))
    rend_uv_mask = torch.tensor(rend_uv_mask.astype(np.float32) / 255.0)[None].to(device)
    
    # get mask for rendering
    mask_region = cv2.imread(config.SKIN_MASK_CROP_PATH)
    mask_region = cv2.resize(mask_region, (uv_crop_w, uv_crop_h))
    mask_region = cv2.cvtColor(mask_region, cv2.COLOR_BGR2GRAY)
    _, mask_region = cv2.threshold(mask_region, 10, 255, cv2.THRESH_BINARY)
    mask_region = mask_region[..., None] / 255.
    
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

    # compute mm
    mm_ret = compute_mm(shape_layer, tex_layer, spec_layer, tri, id, ex, tx, sp, r, tr, s, sh, p, ln, gain, bias)
    uv_ret = compute_uv_render(mm_ret, rast_uv, tri, rend_uv_mask)
    rend_ret = compute_img_render(glctx, mm_ret, uv_ret, tri, uvs, uv_ids, fit_size)

    # for render setting
    v_cam = mm_ret['v_cam']
    v_cam = v_cam / fit_size
    v_cam = torch.cat([v_cam, torch.ones([v_cam.shape[0], v_cam.shape[1], 1]).cuda()], axis=2)
    rast_out, _ = dr.rasterize(glctx, v_cam, tri, resolution=[fit_size, fit_size])
    texc, _ = dr.interpolate(uvs.contiguous(), rast_out, uv_ids.contiguous())
    
    # prepare texture for rendering
    tex_bare = get_uncrop_tex(args.uv_bare_skin_path, mask_region, uv_crop_w, uv_crop_h, unwarp_uv_size, uv_crop_pos).to(device)
    tex_alpha = get_uncrop_tex(args.uv_make_alpha_path, mask_region, uv_crop_w, uv_crop_h, unwarp_uv_size, uv_crop_pos).to(device)
    tex_make = get_uncrop_tex(args.uv_make_blend_path, mask_region, uv_crop_w, uv_crop_h, unwarp_uv_size, uv_crop_pos).to(device)
    tex_diff = get_uncrop_tex(args.uv_diffuse_shading_path, mask_region, uv_crop_w, uv_crop_h, unwarp_uv_size, uv_crop_pos).to(device)
    tex_spec = get_uncrop_tex(args.uv_specular_shading_path, mask_region, uv_crop_w, uv_crop_h, unwarp_uv_size, uv_crop_pos).to(device)
    
    # texture composition
    tex_remake = tex_bare * (1 - tex_alpha) + tex_make
    # or you can read make_base.png, and try 'tex_remake = tex_bare * (1 - tex_alpha) + tex_make_base * tex_alpha'
    # tex_make is equal to tex_make_base * tex_alpha
    
    # add lighting effect to albedo
    tex_compose = tex_remake * tex_diff + tex_spec
    
    # you can replace tex_compose to 'tex_bare' for checking the bare skin face
    img_compose= dr.texture(tex_compose[None], texc, filter_mode='linear')
    img_compose = img_compose * torch.clamp(rast_out[..., -1:], 0, 1)
    save_tensor(img_compose[0], f'{out_dir}/img_compose.png')