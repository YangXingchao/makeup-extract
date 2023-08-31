# -*- coding: utf-8 -*-

import os

# 3DMM definition
MM_PATH = './resources/generic_model.pkl'
LMK_PATH = './resources/landmark_embedding.npy'
MM_SPECULAR_PATH = './resources/albedoModel2020_FLAME_albedoPart.npz'
MM_DIFFUSE_PATH = './resources/FLAME_texture.npz'
MM_MASK_PATH = './resources/FLAME_masks.pkl'
SKIN_MASK_PATH = './resources/skin_mask.png'
SKIN_MASK_CROP_PATH = './resources/skin_mask_crop.png'

n_shape = 200
n_tex = 100
n_exp = 100
n_make = 100
MM_SCALE = 1600

UNWARP_UV_SIZE = 1536

UV_CROP_W = 1152
UV_CROP_H = 1152
UV_CROP_POS = [186, 186]

CROP_RATIO = 1.6
KPT_SIZE = 256
FIT_SIZE = 256
SEG_SIZE = 512
RENDER_SIZE = 1024
UV_TEX_SIZE = 512