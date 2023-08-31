import argparse
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

import config
from networks import CoarseReconsNet
from utils.img_util import save_tensor
from utils.mm_layer import get_mm
from utils.mm_util import compute_mm
from utils.render_util import compute_uv_render, compute_img_render


def save_cropped_sampling_material(uv_tensor, uv_crop_pos, uv_crop_h, uv_crop_w, save_path):
    uv_img = uv_tensor.clone().detach().cpu().numpy()
    uv_img = (255.0 * uv_img).clip(0, 255).astype(np.uint8)
    uv_img = uv_img[uv_crop_pos[1]:uv_crop_pos[1] + uv_crop_h,
             uv_crop_pos[0]:uv_crop_pos[0] + uv_crop_w,
             :]
    cv2.imwrite(save_path, uv_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coarse reconstruction')
    parser.add_argument('--aligned_img_path', default='./results/aligned_img.png', help='Input aligned image')
    parser.add_argument('--segmented_img_path', default='./results/segmented_img.png',
                        help='Input segmented image')
    parser.add_argument('--out_dir', '-o', default='./results/coarse_reconstruction', help='Output directory')
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

    net = CoarseReconsNet(config.n_shape, config.n_exp, config.n_tex, config.n_tex).to(device)
    net.load_state_dict(torch.load('checkpoints/coarse_reconstruction.pkl'))
    net.eval()

    img = cv2.imread(args.aligned_img_path)
    img = cv2.resize(img, (fit_size, fit_size))
    img = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)

    seg_img = cv2.imread(args.segmented_img_path)
    seg_img = cv2.resize(seg_img, (render_size, render_size))
    seg_img = torch.tensor(seg_img.astype(np.float32)).to(device)

    res = net(img)
    mm_ret = compute_mm(shape_layer, tex_layer, spec_layer, tri, res['id'], res['ex'], res['tx'], res['sp'], res['r'],
                        res['tr'], res['s'], res['sh'], res['p'], res['ln'], res['gain'], res['bias'])
    uv_ret = compute_uv_render(mm_ret, rast_uv, tri, rend_uv_mask)
    rend_ret = compute_img_render(glctx, mm_ret, uv_ret, tri, uvs, uv_ids, fit_size)

    v_cam = mm_ret['v_cam']
    v_cam = v_cam / fit_size
    v_cam = torch.cat([v_cam, torch.ones([v_cam.shape[0], v_cam.shape[1], 1]).cuda()], axis=2)
    rast, _ = dr.rasterize(glctx, v_cam, tri, resolution=[render_size, render_size])

    # img rendering
    save_tensor(rend_ret['rncr'][0], f'{out_dir}/rncr.png')
    save_tensor(rend_ret['albe'][0], f'{out_dir}/albe.png')
    save_tensor(rend_ret['diff'][0], f'{out_dir}/diff.png')
    save_tensor(rend_ret['rndr'][0], f'{out_dir}/rndr.png')
    save_tensor(rend_ret['spec'][0], f'{out_dir}/spec.png')
    save_tensor(rend_ret['refl'][0], f'{out_dir}/refl.png')
    save_tensor(rend_ret['rnsr'][0], f'{out_dir}/rnsr.png')

    # sampling uv tex
    r_c_mask = mm_ret['mask'].clone().detach()
    n_mask = mm_ret['n_cam'].clone().detach()
    n_mask = n_mask[:, :, 2].ge(0.02).to(torch.float)[..., None]
    v_cam_uv = v_cam[..., :2].detach().contiguous()
    v_cam_uv = (v_cam_uv + 1) * 0.5

    mask_inv, _ = dr.interpolate(n_mask, rast_uv, tri)
    r_c_mask_inv, _ = dr.interpolate(r_c_mask, rast_uv, tri)
    texc_inv, _ = dr.interpolate(v_cam_uv, rast_uv, tri)
    tex_uv = dr.texture(seg_img[None], texc_inv, filter_mode='linear')

    tex_uv = tex_uv * mask_inv * r_c_mask_inv * rend_uv_mask
    tex_uv_img = tex_uv[0].detach().cpu().numpy().astype(np.uint8)

    # cleaning
    tex_uv_gray = cv2.cvtColor(tex_uv_img, cv2.COLOR_BGR2GRAY)
    _, tex_uv_mask = cv2.threshold(tex_uv_gray, 10, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(
        tex_uv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
    tex_uv_mask = np.zeros_like(tex_uv_mask)
    cv2.drawContours(tex_uv_mask, [max_cnt], -1, color=255, thickness=-1)
    tex_uv_mask = cv2.erode(tex_uv_mask, np.ones((9, 9), np.uint8), iterations=3)
    tex_uv_img = cv2.bitwise_and(tex_uv_img, tex_uv_img, mask=tex_uv_mask)

    contours = cv2.findContours(
        tex_uv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
    tex_uv_mask = np.zeros_like(tex_uv_mask)
    cv2.drawContours(tex_uv_mask, [max_cnt], -1, color=255, thickness=-1)
    tex_uv_img = cv2.bitwise_and(tex_uv_img, tex_uv_img, mask=tex_uv_mask)

    # crop uv tex
    tex_uv_crop = tex_uv_img[uv_crop_pos[1]:uv_crop_pos[1] + uv_crop_h,
                  uv_crop_pos[0]:uv_crop_pos[0] + uv_crop_w,
                  :]
    cv2.imwrite(f'{out_dir}/uv_tex.png', tex_uv_crop)

    save_cropped_sampling_material(uv_ret['norm'][:, :, :, [2, 1, 0]][0] * 0.5 + 0.5, uv_crop_pos, uv_crop_h, uv_crop_w,
                                   f'{out_dir}/uv_norm.png')
    save_cropped_sampling_material(uv_ret['rncr'][0], uv_crop_pos, uv_crop_h, uv_crop_w, f'{out_dir}/uv_rncr.png')
    save_cropped_sampling_material(uv_ret['albe'][0], uv_crop_pos, uv_crop_h, uv_crop_w, f'{out_dir}/uv_albe.png')
    save_cropped_sampling_material(uv_ret['diff'][0], uv_crop_pos, uv_crop_h, uv_crop_w, f'{out_dir}/uv_diff.png')
    save_cropped_sampling_material(uv_ret['rndr'][0], uv_crop_pos, uv_crop_h, uv_crop_w, f'{out_dir}/uv_rndr.png')
    save_cropped_sampling_material(uv_ret['spec'][0], uv_crop_pos, uv_crop_h, uv_crop_w, f'{out_dir}/uv_spec.png')
    save_cropped_sampling_material(uv_ret['refl'][0], uv_crop_pos, uv_crop_h, uv_crop_w, f'{out_dir}/uv_refl.png')
    save_cropped_sampling_material(uv_ret['rnsr'][0], uv_crop_pos, uv_crop_h, uv_crop_w, f'{out_dir}/uv_rnsr.png')

    # save mm params
    id_param = res['id'].detach().cpu().tolist()
    tx_param = res['tx'].detach().cpu().tolist()
    sp_param = res['sp'].detach().cpu().tolist()
    ex_param = res['ex'].detach().cpu().tolist()
    r_param = res['r'].detach().cpu().tolist()
    tr_param = res['tr'].detach().cpu().tolist()
    s_param = res['s'].detach().cpu().tolist()
    sh_param = res['sh'].detach().cpu().tolist()
    p_param = res['p'].detach().cpu().tolist()
    ln_param = res['ln'].detach().cpu().tolist()
    gain_param = res['gain'].detach().cpu().tolist()
    bias_param = res['bias'].detach().cpu().tolist()
    np.savez(f'{out_dir}/mm_param.npz', id=id_param, tx=tx_param, ex=ex_param, sp=sp_param, r=r_param,
             tr=tr_param,
             s=s_param, sh=sh_param, p=p_param, ln=ln_param, gain=gain_param, bias=bias_param)