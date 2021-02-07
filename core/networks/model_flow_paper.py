import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from structures import *
from pytorch_ssim import SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import cv2
from torch.autograd import Variable


class Model_flow(nn.Module):
    def __init__(self, cfg):
        super(Model_flow, self).__init__()
        self.fpyramid = FeaturePyramid()
        self.pwc_model = PWC_tf()
        if cfg.mode == 'depth' or cfg.mode == 'flowposenet':
            # Stage 2 training
            for param in self.fpyramid.parameters():
                param.requires_grad = False
            for param in self.pwc_model.parameters():
                param.requires_grad = False
        
        # hyperparameters
        self.dataset = cfg.dataset
        self.num_scales = cfg.num_scales
        self.flow_consist_alpha = cfg.h_flow_consist_alpha
        self.flow_consist_beta = cfg.h_flow_consist_beta

        print("this is paper method.")



    def get_flow_norm(self, flow, p=2):
        '''
        Inputs:
        flow (bs, 2, H, W)
        '''
        flow_norm = torch.norm(flow, p=p, dim=1).unsqueeze(1) + 1e-12
        return flow_norm

    def get_flow_normalization(self, flow, p=2):
        '''
        Inputs:
        flow (bs, 2, H, W)
        '''
        flow_norm = torch.norm(flow, p=p, dim=1).unsqueeze(1) + 1e-12
        flow_normalization = flow / flow_norm.repeat(1,2,1,1)
        return flow_normalization


    def generate_img_pyramid(self, img, num_pyramid):
        img_h, img_w = img.shape[2], img.shape[3]
        img_pyramid = []
        for s in range(num_pyramid):
            img_new = F.adaptive_avg_pool2d(img, [int(img_h / (2**s)), int(img_w / (2**s))]).data
            img_pyramid.append(img_new)
        return img_pyramid

    def warp_flow_pyramid(self, img_pyramid, flow_pyramid):
        img_warped_pyramid = []
        for img, flow in zip(img_pyramid, flow_pyramid):
            img_warped_pyramid.append(warp_flow(img, flow, use_mask=True))
        return img_warped_pyramid

    def compute_loss_pixel(self, img_pyramid, img_warped_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, occ_mask = img_pyramid[scale], img_warped_pyramid[scale], occ_mask_list[scale]
            divider = occ_mask.mean((1,2,3))
            img_diff = torch.abs((img - img_warped)) * occ_mask.repeat(1,3,1,1)
            loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss

    def compute_loss_pixel_without_mask(self, img_pyramid, img_warped_pyramid):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped = img_pyramid[scale], img_warped_pyramid[scale]
            img_diff = torch.abs((img - img_warped))
            loss_pixel = img_diff.mean((1,2,3)) # (B)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss


    def compute_loss_with_mask(self, diff_list, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            diff, occ_mask = diff_list[scale], occ_mask_list[scale]
            divider = occ_mask.mean((1,2,3))
            img_diff = diff * occ_mask.repeat(1,3,1,1)
            loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss

    def compute_diff_weight(self, img_pyramid_from_l, img_pyramid, img_pyramid_from_r):
        diff_fwd = []
        diff_bwd = []
        weight_fwd = []
        weight_bwd = []
        valid_bwd = []
        valid_fwd = []
        for scale in range(self.num_scales):
            img_from_l, img, img_from_r = img_pyramid_from_l[scale], img_pyramid[scale], img_pyramid_from_r[scale]
            
            valid_pixels_fwd = 1 - (img_from_r == 0).prod(1, keepdim=True).type_as(img_from_r)
            valid_pixels_bwd = 1 - (img_from_l == 0).prod(1, keepdim=True).type_as(img_from_l)

            valid_bwd.append(valid_pixels_bwd)
            valid_fwd.append(valid_pixels_fwd)

            img_diff_l = torch.abs((img-img_from_l)).mean(1, True)
            img_diff_r = torch.abs((img-img_from_r)).mean(1, True)

            diff_cat = torch.cat((img_diff_l, img_diff_r),1)
            weight = 1 - nn.functional.softmax(diff_cat,1)
            weight = Variable(weight.data,requires_grad=False)

            # weight = (weight > 0.48).float()

            weight = 2*torch.exp(-(weight-0.5)**2/0.03)

            weight_bwd.append(torch.unsqueeze(weight[:,0,:,:],1) * valid_pixels_bwd)
            weight_fwd.append(torch.unsqueeze(weight[:,1,:,:],1) * valid_pixels_fwd)

            diff_fwd.append(img_diff_r)
            diff_bwd.append(img_diff_l)
             
        return diff_bwd, diff_fwd, weight_bwd, weight_fwd


    def compute_loss_ssim(self, img_pyramid, img_warped_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, occ_mask = img_pyramid[scale], img_warped_pyramid[scale], occ_mask_list[scale]
            divider = occ_mask.mean((1,2,3))
            occ_mask_pad = occ_mask.repeat(1,3,1,1)
            ssim = SSIM(img * occ_mask_pad, img_warped * occ_mask_pad)
            loss_ssim = torch.clamp((1.0 - ssim) / 2.0, 0, 1).mean((1,2,3))
            loss_ssim = loss_ssim / (divider + 1e-12)
            loss_list.append(loss_ssim[:,None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss



    def gradients(self, img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy

    def cal_grad2_error(self, flow, img):
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
        #error = (w_x * torch.abs(dx)).mean((1,2,3)) + (w_y * torch.abs(dy)).mean((1,2,3))
        return error / 2.0

    def compute_loss_flow_smooth(self, optical_flows, img_pyramid):
        loss_list = []
        for scale in range(self.num_scales):
            flow, img = optical_flows[scale], img_pyramid[scale]
            #error = self.cal_grad2_error(flow, img)
            error = self.cal_grad2_error(flow/20.0, img)
            loss_list.append(error[:,None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss


    def compute_loss_flow_consis(self, fwd_flow_pyramid, bwd_flow_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            fwd_flow, bwd_flow, occ_mask = fwd_flow_pyramid[scale], bwd_flow_pyramid[scale], occ_mask_list[scale]
            fwd_flow_norm = self.get_flow_normalization(fwd_flow)
            bwd_flow_norm = self.get_flow_normalization(bwd_flow)
            bwd_flow_norm = Variable(bwd_flow_norm.data,requires_grad=False)
            occ_mask = 1-occ_mask

            divider = occ_mask.mean((1,2,3))
            
            loss_consis = (torch.abs(fwd_flow_norm+bwd_flow_norm) * occ_mask).mean((1,2,3))
            loss_consis = loss_consis / (divider + 1e-12)
            loss_list.append(loss_consis[:,None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss


    def inference_flow(self, img1, img2):
        img_hw = [img1.shape[2], img1.shape[3]]
        feature_list_1, feature_list_2 = self.fpyramid(img1), self.fpyramid(img2)
        optical_flow = self.pwc_model(feature_list_1, feature_list_2, img_hw)[0]
        return optical_flow
    

    def forward(self, inputs, output_flow=False, use_flow_loss=True, is_second_phase=False):
        images = inputs
        assert (images.shape[1] == 3)
        img_h, img_w = int(images.shape[2] / 3), images.shape[3] 
        imgl, img, imgr = images[:,:,:img_h,:], images[:,:,img_h:2*img_h,:], images[:,:,2*img_h:3*img_h,:]
        batch_size = imgl.shape[0]

        #pdb.set_trace()
        # get the optical flows and reverse optical flows for each pair of adjacent images
        feature_list_l, feature_list, feature_list_r = self.fpyramid(imgl), self.fpyramid(img), self.fpyramid(imgr)
        
        optical_flows_bwd = self.pwc_model(feature_list, feature_list_l, [img_h, img_w])
        #optical_flows_bwd_rev = self.pwc_model(feature_list_l, feature_list, [img_h, img_w])
        optical_flows_fwd = self.pwc_model(feature_list, feature_list_r, [img_h, img_w])
        #optical_flows_fwd_rev = self.pwc_model(feature_list_r, feature_list, [img_h, img_w])


        #cv2.imwrite('./meta/imgl.png', np.transpose(255*imgl[0].cpu().detach().numpy(), [1,2,0]).astype(np.uint8))
        #cv2.imwrite('./meta/img.png', np.transpose(255*img[0].cpu().detach().numpy(), [1,2,0]).astype(np.uint8))
        #cv2.imwrite('./meta/imgr.png', np.transpose(255*imgr[0].cpu().detach().numpy(), [1,2,0]).astype(np.uint8))

        
        loss_pack = {}
        # warp images
        imgl_pyramid = self.generate_img_pyramid(imgl, len(optical_flows_fwd))
        img_pyramid = self.generate_img_pyramid(img, len(optical_flows_fwd))
        imgr_pyramid = self.generate_img_pyramid(imgr, len(optical_flows_fwd))

        img_warped_pyramid_from_l = self.warp_flow_pyramid(imgl_pyramid, optical_flows_bwd)
        #imgl_warped_pyramid_from_ = self.warp_flow_pyramid(img_pyramid, optical_flows_bwd_rev)
        img_warped_pyramid_from_r = self.warp_flow_pyramid(imgr_pyramid, optical_flows_fwd)
        #imgr_warped_pyramid_from_ = self.warp_flow_pyramid(img_pyramid, optical_flows_fwd_rev)



        diff_bwd, diff_fwd, weight_bwd, weight_fwd = self.compute_diff_weight(img_warped_pyramid_from_l, img_pyramid, img_warped_pyramid_from_r)
        loss_pack['loss_pixel'] = self.compute_loss_with_mask(diff_fwd, weight_fwd) + \
            self.compute_loss_with_mask(diff_bwd, weight_bwd)
        
        loss_pack['loss_ssim'] = self.compute_loss_ssim(img_pyramid, img_warped_pyramid_from_r, weight_fwd) + \
            self.compute_loss_ssim(img_pyramid, img_warped_pyramid_from_l,weight_bwd)
        #loss_pack['loss_ssim']  = torch.zeros([2]).to(imgl.get_device()).requires_grad_()

        loss_pack['loss_flow_smooth'] = self.compute_loss_flow_smooth(optical_flows_fwd, img_pyramid)  + \
            self.compute_loss_flow_smooth(optical_flows_bwd, img_pyramid)
        
        loss_pack['loss_flow_consis'] = self.compute_loss_flow_consis(optical_flows_fwd, optical_flows_bwd, weight_fwd)
        # loss_pack['loss_flow_consis'] = torch.zeros([2]).to(imgl.get_device()).requires_grad_()


        return loss_pack
