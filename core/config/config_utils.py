import os, sys

def generate_loss_weights_dict(cfg):
    weight_dict = {}
    weight_dict['loss_pixel'] = 1 - cfg.w_ssim
    weight_dict['loss_ssim'] = cfg.w_ssim
    weight_dict['loss_flow_smooth'] = cfg.w_flow_smooth
    weight_dict['loss_flow_consis'] = cfg.w_flow_consis
    return weight_dict
