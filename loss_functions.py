# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.
# based on github.com/ClementPinard/SfMLearner-Pytorch

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from inverse_warp import inverse_warp, inverse_warp_wmove, flow_warp, pose2flow, pose2flow_wmove, inverse_warp2, compute_fundmental_matrix, compute_interpolation_depth, compute_interpolation_depth_wmove, pose_vec2mat
from ssim import ssim
#from batch_svd import batch_svd
epsilon = 1e-8


def compute_rigid_mask_from_dc(tgt_img, tgt_depths, ref_depths, poses, intrinsics, intrinsics_inv, nlevels, rotation_mode='euler', padding_mode='zeros'):
    def one_scale_one_direction(tgt_depth, ref_depth, pose, intrinsics, intrinsics_inv):
        tgt_depth_interpolated, tgt_depth_computed = compute_interpolation_depth(tgt_depth[:,0], ref_depth, pose, intrinsics, intrinsics_inv, rotation_mode, padding_mode)
        diff = (tgt_depth_interpolated - tgt_depth_computed).abs()
        diff_mean = diff.mean()
        rigid_mask_from_dc = (diff > 3*diff_mean).type(torch.FloatTensor)

        return rigid_mask_from_dc

    rigid_mask_from_dc_bwd = []
    rigid_mask_from_dc_fwd = []   
    for i, ref_depth in enumerate(ref_depths):
        pose = poses[i]
        for j in range(nlevels):
            b, _, h, w = ref_depth[j].size()
            downscale = tgt_img.size(2)/h
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
            rigid_mask_from_dc = one_scale_one_direction(tgt_depths[j],ref_depth[j],pose,intrinsics_scaled,intrinsics_scaled_inv)
            if i == 0:
                rigid_mask_from_dc_bwd.append(rigid_mask_from_dc)
            elif i == 1:
                rigid_mask_from_dc_fwd.append(rigid_mask_from_dc)

    return rigid_mask_from_dc_bwd, rigid_mask_from_dc_fwd


def compute_depth_consis_wmove(object_move, tgt_img, tgt_depths, ref_depths, poses, intrinsics, intrinsics_inv, nlevels, rotation_mode='euler', padding_mode='zeros'):
    def one_scale_one_direction(object_move, tgt_depth, ref_depth, pose, intrinsics, intrinsics_inv):
        tgt_depth_interpolated, tgt_depth_computed = compute_interpolation_depth_wmove(object_move, tgt_depth[:,0], ref_depth, pose, intrinsics, intrinsics_inv, rotation_mode, padding_mode)
        #tgt_depth_interpolated, tgt_depth_computed = compute_interpolation_depth(tgt_depth[:,0], ref_depth, pose, intrinsics, intrinsics_inv, rotation_mode, padding_mode)
        compute_interpolation_depth_wmove
        diff = ((tgt_depth_computed - tgt_depth_interpolated).abs() / (tgt_depth_computed + tgt_depth_interpolated).abs()).clamp(0,1)
        consis_loss = robust_l1(diff)
        return consis_loss

    loss = 0
    if type(object_move) not in [list, tuple]:
        object_move = [object_move]

    for i, ref_depth in enumerate(ref_depths):
        pose = poses[i]
        for j in range(nlevels):
            b, _, h, w = ref_depth[j].size()
            downscale = tgt_img.size(2)/h
            ob_move = object_move[i][j]
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
            loss += one_scale_one_direction(tgt_depths[j],ref_depth[j],pose,intrinsics_scaled,intrinsics_scaled_inv)
    return loss


def compute_depth_consis(tgt_img, tgt_depths, ref_depths, poses, intrinsics, intrinsics_inv, nlevels, rotation_mode='euler', padding_mode='zeros'):
    def one_scale_one_direction(tgt_depth, ref_depth, pose, intrinsics, intrinsics_inv):
        #tgt_depth_interpolated, tgt_depth_computed = compute_interpolation_depth_wmove(object_move, tgt_depth[:,0], ref_depth, pose, intrinsics, intrinsics_inv, rotation_mode, padding_mode)
        tgt_depth_interpolated, tgt_depth_computed = compute_interpolation_depth(tgt_depth[:,0], ref_depth, pose, intrinsics, intrinsics_inv, rotation_mode, padding_mode)
        diff = ((tgt_depth_computed - tgt_depth_interpolated).abs() / (tgt_depth_computed + tgt_depth_interpolated).abs()).clamp(0,1)
        consis_loss = robust_l1(diff)
        return consis_loss

    loss = 0

    for i, ref_depth in enumerate(ref_depths):
        pose = poses[i]
        for j in range(nlevels):
            b, _, h, w = ref_depth[j].size()
            downscale = tgt_img.size(2)/h
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
            loss += one_scale_one_direction(tgt_depths[j],ref_depth[j],pose,intrinsics_scaled,intrinsics_scaled_inv)
    return loss



def compute_consensus_loss_wmove(object_move, tgt_depths, poses, flows, intrinsics, intrinsics_inv, tgt_img, ref_img_fwd, ref_img_bwd, wssim, wrig, nlevels, ws=0.1):
    def one_scale(depth, flow_fwd, flow_bwd, move_fwd, move_bwd, intrinsics_scaled, intrinsics_scaled_inv):
        cam_flow_fwd = pose2flow_wmove(move_fwd, depth.squeeze(1), poses[1], intrinsics_scaled, intrinsics_scaled_inv)
        cam_flow_bwd = pose2flow_wmove(move_bwd, depth.squeeze(1), poses[0], intrinsics_scaled, intrinsics_scaled_inv)
        ##flow_diff_fwd = flow_diff(cam_flow_fwd, flow_fwd)
        #flow_diff_bwd = flow_diff(cam_flow_bwd, flow_bwd)

        # flow_diff_fwd = cam_flow_fwd - flow_fwd
        # flow_diff_bwd = cam_flow_bwd - flow_bwd
        # geometric_loss = (flow_diff_fwd.pow(2).sum(dim=1) + flow_diff_bwd.pow(2).sum(dim=1)).mean()
        
        b, _, h, w = cam_flow_fwd.size()
        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_img_scaled_fwd = nn.functional.adaptive_avg_pool2d(ref_img_fwd, (h, w))
        ref_img_scaled_bwd = nn.functional.adaptive_avg_pool2d(ref_img_bwd, (h, w))

        cam_warped_im_fwd = flow_warp(ref_img_scaled_fwd, cam_flow_fwd)
        cam_warped_im_bwd = flow_warp(ref_img_scaled_bwd, cam_flow_bwd)

        flow_warped_im_fwd = flow_warp(ref_img_scaled_fwd, flow_fwd)
        flow_warped_im_bwd = flow_warp(ref_img_scaled_bwd, flow_bwd)

        valid_pixels_cam_fwd = 1 - (cam_warped_im_fwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_fwd)
        valid_pixels_cam_bwd = 1 - (cam_warped_im_bwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_bwd)
        valid_pixels_cam = logical_or(valid_pixels_cam_fwd, valid_pixels_cam_bwd)  # if one of them is valid, then valid

        valid_pixels_flow_fwd = 1 - (flow_warped_im_fwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_fwd)
        valid_pixels_flow_bwd = 1 - (flow_warped_im_bwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_bwd)
        valid_pixels_flow = logical_or(valid_pixels_flow_fwd, valid_pixels_flow_bwd)  # if one of them is valid, then valid

        valid_pixel = logical_or(valid_pixels_cam, valid_pixels_flow)

        photo_fwd_err = mean_on_mask((cam_warped_im_fwd-flow_warped_im_fwd).abs(), valid_pixel)
        photo_bwd_err = mean_on_mask((cam_warped_im_bwd-flow_warped_im_bwd).abs(), valid_pixel)

        # cam_err_fwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_fwd).mean(1,keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_fwd)).mean(1, keepdim=True))
        # cam_err_bwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_bwd).mean(1,keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_bwd)).mean(1, keepdim=True))
        # cam_err = torch.min(cam_err_fwd, cam_err_bwd) * valid_pixels_cam

        # flow_err = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_fwd).mean(1, keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_fwd)).mean(1, keepdim=True)

        # photometric_loss = (wrig*cam_err - (flow_err+epsilon)).mean()
        return photo_bwd_err+photo_fwd_err

    loss = 0
    for i in range(nlevels):
        depth = tgt_depths[i]
        flow_fwd = flows[1][i]
        flow_bwd = flows[0][i]
        move_fwd = object_move[1][i]
        move_bwd = object_move[0][i]
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        loss += one_scale(depth, flow_fwd, flow_bwd, move_fwd, move_bwd, intrinsics_scaled, intrinsics_scaled_inv)

    return loss.mean()

def compute_consensus_loss(tgt_depths, poses, flows, rigidity_mask, intrinsics, intrinsics_inv, tgt_img, ref_img_fwd, ref_img_bwd, wssim, wrig, nlevels, ws=0.1):
    def one_scale(depth, flow_fwd, flow_bwd,rigidity_mask_scaled, intrinsics_scaled, intrinsics_scaled_inv):
        cam_flow_fwd = pose2flow(depth.squeeze(1), poses[1], intrinsics_scaled, intrinsics_scaled_inv)
        cam_flow_bwd = pose2flow(depth.squeeze(1), poses[0], intrinsics_scaled, intrinsics_scaled_inv)
        # flow_diff_fwd = flow_diff(cam_flow_fwd, flow_fwd)
        # flow_diff_bwd = flow_diff(cam_flow_bwd, flow_bwd)

        # flow_diff_fwd = cam_flow_fwd - flow_fwd
        # flow_diff_bwd = cam_flow_bwd - flow_bwd
        # geometric_loss = (flow_diff_fwd.pow(2).sum(dim=1) + flow_diff_bwd.pow(2).sum(dim=1)).mean()
        
        b, _, h, w = cam_flow_fwd.size()
        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_img_scaled_fwd = nn.functional.adaptive_avg_pool2d(ref_img_fwd, (h, w))
        ref_img_scaled_bwd = nn.functional.adaptive_avg_pool2d(ref_img_bwd, (h, w))
        rigidity_mask_scaled = nn.functional.adaptive_avg_pool2d(rigidity_mask_scaled, (h, w))

        cam_warped_im_fwd = flow_warp(ref_img_scaled_fwd, cam_flow_fwd)
        cam_warped_im_bwd = flow_warp(ref_img_scaled_bwd, cam_flow_bwd)

        flow_warped_im_fwd = flow_warp(ref_img_scaled_fwd, flow_fwd)
        flow_warped_im_bwd = flow_warp(ref_img_scaled_bwd, flow_bwd)

        valid_pixels_cam_fwd = 1 - (cam_warped_im_fwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_fwd)
        valid_pixels_cam_bwd = 1 - (cam_warped_im_bwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_bwd)
        valid_pixels_cam = logical_or(valid_pixels_cam_fwd, valid_pixels_cam_bwd)  # if one of them is valid, then valid

        valid_pixels_flow_fwd = 1 - (flow_warped_im_fwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_fwd)
        valid_pixels_flow_bwd = 1 - (flow_warped_im_bwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_bwd)
        valid_pixels_flow = logical_or(valid_pixels_flow_fwd, valid_pixels_flow_bwd)  # if one of them is valid, then valid

        valid_pixel = logical_and(valid_pixels_cam, valid_pixels_flow)

        photo_fwd_err = mean_on_mask((cam_warped_im_fwd-flow_warped_im_fwd).abs(), valid_pixel*rigidity_mask_scaled)
        photo_bwd_err = mean_on_mask((cam_warped_im_bwd-flow_warped_im_bwd).abs(), valid_pixel*rigidity_mask_scaled)

        # cam_err_fwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_fwd).mean(1,keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_fwd)).mean(1, keepdim=True))
        # cam_err_bwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_bwd).mean(1,keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_bwd)).mean(1, keepdim=True))
        # cam_err = torch.min(cam_err_fwd, cam_err_bwd) * valid_pixels_cam

        # flow_err = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_fwd).mean(1, keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_fwd)).mean(1, keepdim=True)

        # photometric_loss = (wrig*cam_err - (flow_err+epsilon)).mean()
        # #return geometric_loss+photometric_loss
        #return flow_diff_fwd.mean()+flow_diff_bwd.mean()+photo_bwd_err+photo_fwd_err, valid_pixel
        return photo_bwd_err+photo_fwd_err

    loss = 0
    for i in range(nlevels):
        depth = tgt_depths[i]
        rigidity_mask_one_scale = rigidity_mask[i]
        flow_fwd = flows[1][i]
        flow_bwd = flows[0][i]
        # move_fwd = object_move[i][:,3:6]
        # move_bwd = object_move[i][:,0:3]
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
        
        loss += one_scale(depth, flow_fwd, flow_bwd, rigidity_mask_one_scale, intrinsics_scaled, intrinsics_scaled_inv)

    return loss.mean()


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum()
    return mean_value

def spatial_normalize(disp):
    _mean = disp.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    disp = disp / _mean
    return disp

def robust_l1(x, q=0.5, eps=1e-3):
    x = torch.pow((x.pow(2) + eps*eps), q)
    x = x.mean()
    return x

def robust_l1_per_pix(x, q=0.5, eps=1e-3):
    x = torch.pow((x.pow(2) + eps*eps), q)
    return x

def photometric_flow_loss(tgt_img, ref_imgs, flows, lambda_oob=0, qch=0.38, wssim=0.0):
    def one_scale(flows):
        #assert(explainability_mask is None or flows[0].size()[2:] == explainability_mask.size()[2:])
        assert(len(flows) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = flows[0].size()

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        loss = 0.0
        for i, ref_img in enumerate(ref_imgs_scaled):
            current_flow = flows[i]
            ref_img_warped = flow_warp(ref_img, current_flow)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) 
            if wssim:
                ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) 
                reconstruction_loss = (1- wssim)*robust_l1_per_pix(diff.mean(1, True), q=qch)*valid_pixels + wssim*ssim_loss.mean(1, True) 
            else:
                reconstruction_loss = robust_l1_per_pix(diff.mean(1, True), q=qch)*valid_pixels
            loss += reconstruction_loss.sum()/valid_pixels.sum()

        return loss

    if type(flows[0]) not in [tuple, list]:
        # if explainability_mask is not None:
        #     explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    weight = 1.0
    for i in range(len(flows[0])):
        flow_at_scale = [uv[i] for uv in flows]
        loss += weight*one_scale(flow_at_scale)
        weight /= 2.3
    return loss


def photometric_flow_min_loss(tgt_img, ref_imgs, flows, lambda_oob=0, qch=0.5, wssim=0.0):
    def one_scale(flows):
        #assert(explainability_mask is None or flows[0].size()[2:] == explainability_mask.size()[2:])
        assert(len(flows) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = flows[0].size()

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        reconstruction_loss_all = []
        ssim_loss = 0.0
        for i, ref_img in enumerate(ref_imgs_scaled):
            current_flow = flows[i]
            ref_img_warped = flow_warp(ref_img, current_flow)
            # valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) 
             
            if wssim:
                ssim_loss += wssim*(1 - ssim(tgt_img_scaled, ref_img_warped)).mean()
            
            reconstruction_loss = robust_l1_per_pix(diff.mean(1, True), q=qch)

            reconstruction_loss_all.append(reconstruction_loss)

        reconstruction_loss = torch.cat(reconstruction_loss_all,1)
        reconstruction_weight = reconstruction_loss 
        # reconstruction_loss_min,_ = reconstruction_loss.min(1,keepdim=True)
        # reconstruction_loss_min = reconstruction_loss_min.repeat(1,2,1,1)
        # loss_weight = reconstruction_loss_min/reconstruction_loss
        # loss_weight = torch.pow(loss_weight,4)

        loss_weight = 1 - torch.nn.functional.softmax(reconstruction_weight, 1)
        loss_weight = Variable(loss_weight.data,requires_grad=False)
        loss = reconstruction_loss*loss_weight
        # loss = torch.mean(loss,3)
        # loss = torch.mean(loss,2)
        # loss = torch.mean(loss,0)
        return loss.sum()/loss_weight.sum()+ssim_loss

    if type(flows[0]) not in [tuple, list]:
        # if explainability_mask is not None:
        #     explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    weight = 1.0
    for i in range(len(flows[0])):
    #for i in range(1):
        flow_at_scale = [uv[i] for uv in flows]
        loss += weight*one_scale(flow_at_scale)
        weight /= 2.3

    return loss

def photometric_flow_gradient_loss(tgt_img, ref_imgs, flows, lambda_oob=0, qch=0.38, wssim=0.0):
    def one_scale(flows):
        #assert(explainability_mask is None or flows[0].size()[2:] == explainability_mask.size()[2:])
        assert(len(flows) == len(ref_imgs))

        reconstruction_loss = 0
        _, _, h, w = flows[0].size()

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        reconstruction_loss_all = 0.0
        for i, ref_img in enumerate(ref_imgs_scaled):
            current_flow = flows[i]
            ref_img_warped = flow_warp(ref_img, current_flow)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            reconstruction_loss = gradient_photometric_loss(tgt_img_scaled, ref_img_warped, qch)*valid_pixels[:,:,:-1,:-1]
            # reconstruction_loss = gradient_photometric_all_direction_loss(tgt_img_scaled, ref_img_warped, qch)*valid_pixels[:,:,1:-1,1:-1]

            reconstruction_loss_all += reconstruction_loss.sum()/valid_pixels[:,:,:-1,:-1].sum()

        return reconstruction_loss_all

    if type(flows[0]) not in [tuple, list]:
        # if explainability_mask is not None:
        #     explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    weight = 1.0
    for i in range(len(flows[0])):
        flow_at_scale = [uv[i] for uv in flows]
        loss += weight*one_scale(flow_at_scale)
        weight /= 2.3

    return loss

def scale_weight(x,m,E):
    x = 1/(1+torch.pow(m/x,8))
    return x

def photometric_flow_gradient_min_loss(tgt_img, ref_imgs, flows, lambda_oob=0, qch=0.5, wssim=0.0, wconsis=0.0):
    def one_scale(flows):
        assert(len(flows) == len(ref_imgs))

        # reconstruction_loss = 0
        b, _, h, w = flows[0].size()

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        reconstruction_loss_all = []
        reconstruction_weight_all = []
        # consistancy_loss_all = []
        ssim_loss = 0.0
        for i, ref_img in enumerate(ref_imgs_scaled):
            current_flow = flows[i]
            ref_img_warped = flow_warp(ref_img, current_flow)
            diff = (tgt_img_scaled - ref_img_warped)

            if wssim:
                ssim_loss += wssim*(1 - ssim(tgt_img_scaled, ref_img_warped)).mean()

            # reconstruction_loss = gradient_photometric_loss(tgt_img_scaled, ref_img_warped, qch)
            reconstruction_loss = gradient_photometric_all_direction_loss(tgt_img_scaled, ref_img_warped, qch)
            reconstruction_weight = robust_l1_per_pix(diff.mean(1, True), q=qch)
            # reconstruction_weight = reconstruction_loss
            reconstruction_loss_all.append(reconstruction_loss)
            reconstruction_weight_all.append(reconstruction_weight)
            # consistancy_loss_all.append(reconstruction_loss)


        reconstruction_loss = torch.cat(reconstruction_loss_all,1)
        reconstruction_weight = torch.cat(reconstruction_weight_all,1)
        # consistancy_loss = torch.cat(consistancy_loss_all,1)

        # reconstruction_weight_min,_ = reconstruction_weight.min(1,keepdim=True)
        # reconstruction_weight_min = reconstruction_weight_min.repeat(1,2,1,1)
        # reconstruction_weight_sum = reconstruction_weight.sum(1,keepdim=True)
        # reconstruction_weight_sum = reconstruction_weight_sum.repeat(1,2,1,1)

        # consistancy_loss = consistancy_loss[:,0,:,:]-consistancy_loss[:,1,:,:]
        # consistancy_loss = wconsis*torch.mean(torch.abs(consistancy_loss))

        # loss_weight = reconstruction_weight_min/(reconstruction_weight)
        # loss_weight = reconstruction_weight/reconstruction_weight_sum
        loss_weight = 1 - torch.nn.functional.softmax(reconstruction_weight, 1)
        # loss_weight = (loss_weight >= 0.4).type_as(reconstruction_loss)
        # print(loss_weight.size())
        # loss_weight = loss_weight[:,:,:-1,:-1]
        loss_weight = loss_weight[:,:,1:-1,1:-1]
        # loss_weight = scale_weight(loss_weight,0.3,10)

        # # loss_weight = torch.pow(loss_weight,4)
        loss_weight = Variable(loss_weight.data,requires_grad=False)
        loss = reconstruction_loss*loss_weight
        # loss, _ = torch.min(reconstruction_loss, dim=1)
        # # loss = torch.mean(loss,3)
        # # loss = torch.mean(loss,2)
        # # loss = torch.mean(loss,0)
        # loss, _ = torch.min(reconstruction_loss, dim=1)
        loss = loss.sum()/loss_weight.sum()
        return loss+ssim_loss, loss_weight

    if type(flows[0]) not in [tuple, list]:
        # if explainability_mask is not None:
        #     explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    weight = 1.0
    loss_weight = []
    for i in range(len(flows[0])):
    #for i in range(1):
        flow_at_scale = [uv[i] for uv in flows]
        
        loss_scale, loss_weight_scale = one_scale(flow_at_scale)
        loss += weight*loss_scale
        loss_weight.append(loss_weight_scale)
        weight /= 2.3

    return loss, loss_weight

def flow_velocity_consis_loss(flows):
    def one_scale(flow):
        flow_fwd = flow[1]
        flow_bwd = flow[0]
        flow_bwd_fix = Variable(flow_bwd.data, requires_grad=False)
        vc_loss = robust_l1(flow_fwd+flow_bwd_fix,q=0.5)
        # vc_loss = robust_l1(flow_fwd+flow_bwd,q=0.5)
        return vc_loss

    if type(flows[0]) not in [tuple, list]:
        # if explainability_mask is not None:
        #     explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    for i in range(len(flows[0])):
    # for i in range(1):
        flow_at_scale = [uv[i] for uv in flows]
        loss += one_scale(flow_at_scale)

    return loss

#def consistancy_loss(tgt_img, ref_imgs, flows, qch=0.38)

def photometric_reconstruction_loss_wmove(object_move, tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, poses,  nlevels, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.5):
    def one_scale(object_move, depth, occ_masks):
        assert(len(poses) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        weight = 1.

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = poses[i]

            #ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            ref_img_warped = inverse_warp_wmove(object_move[i], ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

            assert((oob_normalization_const == oob_normalization_const).item() == 1)

            diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
            ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss +=  (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            assert((reconstruction_loss == reconstruction_loss).item() == 1)
            #weight /= 2.83
        return reconstruction_loss

    if type(depth) not in [list, tuple]:
        depth = [depth]
    if type(object_move) not in [list, tuple]:
        object_move = [object_move]

    loss = 0
    # for i in range(nlevels):
    #     occ_masks = depth_occlusion_masks(depth[i], poses, intrinsics, intrinsics_inv)
    #     loss += one_scale([object_move[0][i],object_move[1][i]], depth[i], occ_masks)
    return loss


def photometric_reconstruction_loss(p_mask, tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, poses,  nlevels, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.85):
    def one_scale(depth):
        # depth size : B 1 256 832
        assert(len(poses) == len(ref_imgs))
        reconstruction_loss_all = []
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = poses[i]
            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            #valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) 
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) 

            diff = diff *(p_mask).expand_as(diff)
            ssim_loss = ssim_loss*(p_mask).expand_as(ssim_loss)

            reconstruction_loss = (1- wssim)*robust_l1_per_pix(diff.mean(1, True), q=qch) + wssim*ssim_loss.mean(1, True)
            reconstruction_loss_all.append(reconstruction_loss)

        reconstruction_loss =  torch.cat(reconstruction_loss_all, 1)
        loss, _ = torch.min(reconstruction_loss, dim=1)
        return loss.mean()

    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss = 0
    for i in range(nlevels):
        #occ_masks = depth_occlusion_masks(depth[i], poses, intrinsics, intrinsics_inv)
        loss += one_scale(depth[i])
    return loss


def gaussian_explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        loss += torch.exp(-torch.mean((mask_scaled-0.5).pow(2))/0.15)
    return loss


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = Variable(torch.ones(1)).expand_as(mask_scaled).type_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def logical_or(a, b):
    return 1 - (1 - a)*(1 - b)

def logical_and(a, b):
    return a*b


def compute_masked_motion(motion_bwd, motion_fwd, mask):
    motion_masked_bwd = []
    for i in range(len(motion_bwd)):
        motion_masked_bwd_one_scale = motion_bwd[i]*mask[i].expand_as(motion_bwd[i])
        motion_masked_bwd.append(motion_masked_bwd_one_scale)

    motion_masked_fwd = []
    for i in range(len(motion_fwd)):
        motion_masked_fwd_one_scale = motion_fwd[i]*mask[i].expand_as(motion_fwd[i])
        motion_masked_fwd.append(motion_masked_fwd_one_scale)

    return motion_masked_bwd, motion_masked_fwd


def edge_aware_smoothness_per_pixel(img, pred):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    pred_gradients_x = gradient_x(pred)
    pred_gradients_y = gradient_y(pred)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = torch.abs(pred_gradients_x) * weights_x
    smoothness_y = torch.abs(pred_gradients_y) * weights_y
    #import ipdb; ipdb.set_trace()
    return smoothness_x + smoothness_y


def gradient_photometric_loss(img1, img2, qch=0.5):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    gx_img1 = gradient_x(img1)
    gx_img2 = gradient_x(img2)
    gy_img1 = gradient_y(img1)
    gy_img2 = gradient_y(img2)

    diffx = (gx_img1 - gx_img2)[:,:,:,:-1]
    diffy = (gy_img1 - gy_img2)[:,:,:-1,:]

    loss = robust_l1_per_pix(diffx.mean(1, True), q=qch) + robust_l1_per_pix(diffy.mean(1, True), q=qch)
    return loss

def gradient_photometric_all_direction_loss(img1, img2, qch=0.5):
    def gradient_x(img):
      gx = img[:,:,1:-1,1:-1] - img[:,:,2:,1:-1]
      return gx

    def gradient_24(img):
      gx = img[:,:,1:-1,1:-1] - img[:,:,2:,2:]
      return gx
    
    def gradient_y(img):
      gy = img[:,:,1:-1,1:-1] - img[:,:,1:-1,2:]
      return gy

    def gradient_13(img):
      gy = img[:,:,1:-1,1:-1] - img[:,:,2:,:-2]
      return gy

    gx_img1 = gradient_x(img1)
    gx_img2 = gradient_x(img2)
    gy_img1 = gradient_y(img1)
    gy_img2 = gradient_y(img2)

    g13_img1 = gradient_13(img1)
    g13_img2 = gradient_13(img2)
    g24_img1 = gradient_24(img1)
    g24_img2 = gradient_24(img2)

    diffx = (gx_img1 - gx_img2)
    diffy = (gy_img1 - gy_img2)

    diff13 = (g13_img1 - g13_img2)
    diff24 = (g24_img1 - g24_img2)

    loss = robust_l1_per_pix(diffx.mean(1, True), q=qch) + robust_l1_per_pix(diffy.mean(1, True), q=qch)
    loss += robust_l1_per_pix(diff13.mean(1, True), q=qch) + robust_l1_per_pix(diff24.mean(1, True), q=qch)

    return loss

def edge_aware_smoothness_loss(img, pred_disp):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    def get_edge_smoothness(img, pred):
      pred_gradients_x = gradient_x(pred)
      pred_gradients_y = gradient_y(pred)

      image_gradients_x = gradient_x(img)
      image_gradients_y = gradient_y(img)

      weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
      weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

      smoothness_x = torch.abs(pred_gradients_x) * weights_x
      smoothness_y = torch.abs(pred_gradients_y) * weights_y
      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.0

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        # mean_disp = scaled_disp.mean(2, True).mean(3, True)
        # norm_disp = scaled_disp / (mean_disp + 1e-7)
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += weight*get_edge_smoothness(scaled_img, scaled_disp)
        # weight /= 4   # 2sqrt(2)
        weight /= 2.3   # 2sqrt(2)


    return loss

def edge_aware_smoothness_loss_change_weight(img, pred_disp, alpha):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    def get_edge_smoothness(img, pred):
      pred_gradients_x = gradient_x(pred)
      pred_gradients_y = gradient_y(pred)

      image_gradients_x = gradient_x(img)
      image_gradients_y = gradient_y(img)

      weights_x = torch.exp(-alpha*torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
      weights_y = torch.exp(-alpha*torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

      smoothness_x = torch.abs(pred_gradients_x) * weights_x
      smoothness_y = torch.abs(pred_gradients_y) * weights_y
      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.0

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        # mean_disp = scaled_disp.mean(2, True).mean(3, True)
        # norm_disp = scaled_disp / (mean_disp + 1e-7)
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += weight*get_edge_smoothness(scaled_img, scaled_disp)
        # weight /= 4   # 2sqrt(2)
        weight /= 2.3   # 2sqrt(2)


    return loss


def edge_aware_smoothness_second_order_loss(img, pred_disp):
    def gradient_x_up(img):
      gx = img[:,:,:-2,:] - img[:,:,1:-1,:]
      return gx

    def gradient_x_down(img):
      gx = img[:,:,1:-1,:] - img[:,:,2:,:]
      return gx

    def gradient_y_up(img):
      gy = img[:,:,:,:-2] - img[:,:,:,1:-1]
      return gy

    def gradient_y_down(img):
      gy = img[:,:,:,1:-1] - img[:,:,:,2:]
      return gy

    def get_edge_smoothness(img, pred):
      pred_gradients_x_up = gradient_x_up(pred)
      pred_gradients_x_down = gradient_x_down(pred)
      pred_gradients_y_up = gradient_y_up(pred)
      pred_gradients_y_down = gradient_y_down(pred)

      image_gradients_x_up = gradient_x_up(pred)
      image_gradients_x_down = gradient_x_down(pred)
      image_gradients_y_up = gradient_y_up(pred)
      image_gradients_y_down = gradient_y_down(pred)

      weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x_down), 1, keepdim=True))*torch.exp(-torch.mean(torch.abs(image_gradients_x_up), 1, keepdim=True))
      weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y_down), 1, keepdim=True))*torch.exp(-torch.mean(torch.abs(image_gradients_y_up), 1, keepdim=True))

      smoothness_x = robust_l1_per_pix(pred_gradients_x_up-pred_gradients_x_down) * weights_x
      smoothness_y = robust_l1_per_pix(pred_gradients_y_up-pred_gradients_y_down) * weights_y

      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.0

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        # mean_disp = scaled_disp.mean(2, True).mean(3, True)
        # norm_disp = scaled_disp / (mean_disp + 1e-7)
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += weight*get_edge_smoothness(scaled_img, scaled_disp)
        weight /= 2.3   # 2sqrt(2)
        # weight /= 4   # 2sqrt(2)

    return loss


def edge_aware_smoothness_second_order_loss_change_weight(img, pred_disp,alpha):
    def gradient_x_up(img):
      gx = img[:,:,:-2,:] - img[:,:,1:-1,:]
      return gx

    def gradient_x_down(img):
      gx = img[:,:,1:-1,:] - img[:,:,2:,:]
      return gx

    def gradient_y_up(img):
      gy = img[:,:,:,:-2] - img[:,:,:,1:-1]
      return gy

    def gradient_y_down(img):
      gy = img[:,:,:,1:-1] - img[:,:,:,2:]
      return gy

    def get_edge_smoothness(img, pred, alpha):
    #   pred_gradients_x_up = gradient_x_up(pred)
      pred_gradients_x_down = gradient_x_down(pred)
    #   pred_gradients_y_up = gradient_y_up(pred)
      pred_gradients_y_down = gradient_y_down(pred)

    #   image_gradients_x_up = gradient_x_up(pred)
      image_gradients_x_down = gradient_x_down(pred)
    #   image_gradients_y_up = gradient_y_up(pred)
      image_gradients_y_down = gradient_y_down(pred)

    #   weights_x = torch.exp(-alpha*torch.mean(torch.abs(image_gradients_x_down), 1, keepdim=True))
    #   weights_y = torch.exp(-alpha*torch.mean(torch.abs(image_gradients_y_down), 1, keepdim=True))
      weights_x = torch.exp(-alpha*torch.abs(image_gradients_x_down))
      weights_y = torch.exp(-alpha*torch.abs(image_gradients_y_down))
      smoothness_x = pred_gradients_x_down.pow(2)*weights_x
      smoothness_y = pred_gradients_y_down.pow(2)*weights_y

      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.0

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += weight*get_edge_smoothness(scaled_img, scaled_disp, alpha)
        weight /= 2.3   # 2sqrt(2)
        # weight /= 4
    return loss

def edge_aware_smoothness_second_all_direction_loss(img, pred_disp,alpha):
    def gradient_x(img):
      gx = img[:,:,1:-1,1:-1] - img[:,:,2:,1:-1]
      return gx

    def gradient_24(img):
      gx = img[:,:,1:-1,1:-1] - img[:,:,2:,2:]
      return gx
    
    def gradient_y(img):
      gy = img[:,:,1:-1,1:-1] - img[:,:,1:-1,2:]
      return gy

    def gradient_13(img):
      gy = img[:,:,1:-1,1:-1] - img[:,:,2:,:-2]
      return gy


    def get_edge_smoothness(img, pred, alpha):

      gx_img = gradient_x(img)
      gx_pred = gradient_x(pred)
      
      gy_img = gradient_y(img)
      gy_pred = gradient_y(pred)

      g13_img = gradient_13(img)
      g13_pred = gradient_13(pred)
      
      g24_img = gradient_24(img)
      g24_pred = gradient_24(pred)


    #   weights_x = torch.exp(-alpha*torch.mean(torch.abs(image_gradients_x_down), 1, keepdim=True))
    #   weights_y = torch.exp(-alpha*torch.mean(torch.abs(image_gradients_y_down), 1, keepdim=True))
      weight_x = torch.exp(-alpha*torch.mean(torch.abs(gx_img), 1, keepdim=True))
      weight_y = torch.exp(-alpha*torch.mean(torch.abs(gy_img), 1, keepdim=True))
      weight_13 = torch.exp(-alpha*torch.mean(torch.abs(g13_img), 1, keepdim=True))
      weight_24 = torch.exp(-alpha*torch.mean(torch.abs(g24_img), 1, keepdim=True))

      smoothness_x  = gx_pred.pow(2)*weight_x
      smoothness_y  = gy_pred.pow(2)*weight_y
      smoothness_13 = g13_pred.pow(2)*weight_13
      smoothness_24 = g24_pred.pow(2)*weight_24


      return torch.mean(smoothness_x) + torch.mean(smoothness_y) + torch.mean(smoothness_13) + torch.mean(smoothness_24)

    loss = 0
    weight = 1.0

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += weight*get_edge_smoothness(scaled_img, scaled_disp, alpha)
        weight /= 2.3   # 2sqrt(2)
        # weight /= 4
    return loss

def smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_disp) not in [tuple, list]:
        pred_disp = [pred_disp]

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        dx, dy = gradient(scaled_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # 2sqrt(2)
    return loss


def photometric_mask(origin_img):
    def gradient_mask(pred):
        b, h, w = pred.size()
        D_dy = pred[:, 1:] - pred[:, :-1]
        D_dx = pred[:, :, 1:] - pred[:, :, :-1]
        mask_x = (D_dx == 0).type(torch.FloatTensor)
        mask_x = nn.functional.adaptive_avg_pool2d(mask_x, (h, w))
        mask_y = (D_dy == 0).type(torch.FloatTensor)
        mask_y = nn.functional.adaptive_avg_pool2d(mask_y, (h, w))
        return ((mask_x+mask_y) == 2).type(torch.FloatTensor)

    # b, _, h, w = cam_flow_fwd.size()
    # tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))


    mask_1 = gradient_mask(origin_img[:,0,:,:])
    mask_2 = gradient_mask(origin_img[:,1,:,:])
    mask_3 = gradient_mask(origin_img[:,2,:,:])

    mask = mask_1+mask_2+mask_3

    return ((mask == 3).type(torch.FloatTensor)).unsqueeze(dim=1)

def compute_valid_area(tgt_depths, poses, flows, intrinsics, intrinsics_inv, tgt_img, ref_imgs, nlevels, rotation_mode='euler', padding_mode='zeros'):
    def one_scale(depth, flow_fwd, flow_bwd):

        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
        ref_img_scaled_fwd = nn.functional.adaptive_avg_pool2d(ref_imgs[1], (h, w))
        ref_img_scaled_bwd = nn.functional.adaptive_avg_pool2d(ref_imgs[0], (h, w))

        depth_warped_im_fwd = inverse_warp(ref_img_scaled_fwd, depth[:,0], poses[1], intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
        depth_warped_im_bwd = inverse_warp(ref_img_scaled_bwd, depth[:,0], poses[0], intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
        valid_pixels_depth_fwd = 1 - (depth_warped_im_fwd == 0).prod(1, keepdim=True).type_as(depth_warped_im_fwd)
        valid_pixels_depth_bwd = 1 - (depth_warped_im_bwd == 0).prod(1, keepdim=True).type_as(depth_warped_im_bwd)
        valid_pixels_depth = logical_and(valid_pixels_depth_fwd, valid_pixels_depth_bwd)  # if one of them is valid, then valid

        flow_warped_im_fwd = flow_warp(ref_img_scaled_fwd, flow_fwd)
        flow_warped_im_bwd = flow_warp(ref_img_scaled_bwd, flow_bwd)

        valid_pixels_flow_fwd = 1 - (flow_warped_im_fwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_fwd)
        valid_pixels_flow_bwd = 1 - (flow_warped_im_bwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_bwd)
        valid_pixels_flow = logical_and(valid_pixels_flow_fwd, valid_pixels_flow_bwd)  # if one of them is valid, then valid

        valid_pixel = logical_or(valid_pixels_depth, valid_pixels_flow)

        return valid_pixel

    valid_area = []
    for i in range(nlevels):
        depth = tgt_depths[i]
        flow_fwd = flows[1][i]
        flow_bwd = flows[0][i]
        valid_area.append(one_scale(depth, flow_fwd, flow_bwd))

    return valid_area

def compute_rigidity_mask(tgt_depth,poses,flow_fwd,flow_bwd,intrinsics_var,intrinsics_inv_var,valid_area):
    def one_scale(depth_scale, flow_fwd_scale, flow_bwd_scale, valid_area_scale):
        flow_cam_fwd = pose2flow(depth_scale.squeeze(1),poses[1],intrinsics_var,intrinsics_inv_var)
        flow_cam_bwd = pose2flow(depth_scale.squeeze(1),poses[0],intrinsics_var,intrinsics_inv_var)

        # flow_diff_fwd = flow_diff(flow_cam_fwd,flow_fwd_scale)
        # flow_diff_bwd = flow_diff(flow_cam_bwd,flow_bwd_scale)
        flow_diff_fwd = (flow_cam_fwd-flow_fwd_scale).abs()*valid_area_scale
        flow_diff_bwd = (flow_cam_bwd-flow_bwd_scale).abs()*valid_area_scale

        flow_diff_mean_fwd_u = flow_diff_fwd[:,0].mean(2,keepdim=True).mean(1,keepdim=True)
        flow_diff_mean_fwd_v = flow_diff_fwd[:,1].mean(2,keepdim=True).mean(1,keepdim=True)
        flow_diff_mean_bwd_u = flow_diff_bwd[:,0].mean(2,keepdim=True).mean(1,keepdim=True)
        flow_diff_mean_bwd_v = flow_diff_bwd[:,1].mean(2,keepdim=True).mean(1,keepdim=True)

        motion_mask_fwd_u = torch.gt(flow_diff_fwd[:,0],2*flow_diff_mean_fwd_u).type_as(flow_diff_fwd)
        motion_mask_fwd_v = torch.gt(flow_diff_fwd[:,1],2*flow_diff_mean_fwd_v).type_as(flow_diff_fwd)
        motion_mask_bwd_u = torch.gt(flow_diff_bwd[:,0],2*flow_diff_mean_bwd_u).type_as(flow_diff_fwd)
        motion_mask_bwd_v = torch.gt(flow_diff_bwd[:,1],2*flow_diff_mean_bwd_v).type_as(flow_diff_fwd)

        motion_mask_fwd = logical_or(motion_mask_fwd_u,motion_mask_fwd_v)
        motion_mask_bwd = logical_or(motion_mask_bwd_u,motion_mask_bwd_v)

        motion_mask = logical_and(motion_mask_fwd,motion_mask_bwd)
        return motion_mask

    depth_scale = tgt_depth[0]
    flow_fwd_scale = flow_fwd[0]
    flow_bwd_scale = flow_bwd[0]
    valid_area_scale = valid_area[0]
    motion_mask = one_scale(depth_scale,flow_fwd_scale,flow_bwd_scale,valid_area_scale)
    rigidity_mask = 1-Variable(motion_mask.data,requires_grad=False)

    return rigidity_mask.unsqueeze(1)


def flow_diff(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    diff = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    return diff.unsqueeze(dim=1)


def compute_epe(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    if nc == 3:
        valid = gt[:,2,:,:]
        epe = epe * valid
        avg_epe = epe.sum()/(valid.sum() + epsilon)
    else:
        avg_epe = epe.sum()/(bs*h_gt*w_gt)

    if type(avg_epe) == Variable: avg_epe = avg_epe.data

    return avg_epe.item()

def outlier_err(gt, pred, tau=[3,0.05]):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt, valid_gt = gt[:,0,:,:], gt[:,1,:,:], gt[:,2,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    epe = epe * valid_gt

    F_mag = torch.sqrt(torch.pow(u_gt, 2)+ torch.pow(v_gt, 2))
    E_0 = (epe > tau[0]).type_as(epe)
    E_1 = ((epe / (F_mag+epsilon)) > tau[1]).type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    #n_err   = length(find(F_val & E>tau(1) & E./F_mag>tau(2)));
    #n_total = length(find(F_val));
    f_err = n_err.sum()/(valid_gt.sum() + epsilon);
    if type(f_err) == Variable: f_err = f_err.data
    return f_err.item()

def compute_all_epes(gt, rigid_pred, non_rigid_pred, rigidity_mask, THRESH=0.5):
    _, _, h_pred, w_pred = rigid_pred.size()
    _, _, h_gt, w_gt = gt.size()
    rigidity_pred_mask = nn.functional.upsample(rigidity_mask, size=(h_pred, w_pred), mode='bilinear')
    rigidity_gt_mask = nn.functional.upsample(rigidity_mask, size=(h_gt, w_gt), mode='bilinear')

    non_rigid_pred = (rigidity_pred_mask<=THRESH).type_as(non_rigid_pred).expand_as(non_rigid_pred) * non_rigid_pred
    rigid_pred = (rigidity_pred_mask>THRESH).type_as(rigid_pred).expand_as(rigid_pred) * rigid_pred
    total_pred = non_rigid_pred + rigid_pred

    gt_non_rigid = (rigidity_gt_mask<=THRESH).type_as(gt).expand_as(gt) * gt
    gt_rigid = (rigidity_gt_mask>THRESH).type_as(gt).expand_as(gt) * gt

    all_epe = compute_epe(gt, total_pred)
    rigid_epe = compute_epe(gt_rigid, rigid_pred)
    non_rigid_epe = compute_epe(gt_non_rigid, non_rigid_pred)
    outliers = outlier_err(gt, total_pred)

    return [all_epe, rigid_epe, non_rigid_epe, outliers]


def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

def compute_epipolar_loss(pose,flow,intrinsics):
    #check_sizes(flow, 'flow', 'B2HW')

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    zeros = v.detach()*0
    ones = zeros.detach()+1

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]
    
    fundmental_mat = compute_fundmental_matrix(pose, intrinsics)

    X = grid_x + u
    Y = grid_y + v

    left = torch.stack((grid_x,grid_y,ones), dim=1).view(bs,3,h*w)
    right = torch.stack((X,Y,ones), dim=1).view(bs,3,h*w) # print shape [B 2 H W]

    loss_meta = torch.transpose(right,1,2) @ fundmental_mat
    loss = loss_meta*(torch.transpose(left,1,2))
    return robust_l1(loss)

def grid_sample(flow, scale, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        flow: flow map of the target image -- [B, 2, H, W]
    Returns:
        Source image warped to the target image plane
    """

    bs, _, h, w = flow.size()
    
    w = w//scale
    h = h//scale

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).expand(bs,h,w).cuda().float()
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).expand(bs,h,w).cuda().float()

    X = 2*(grid_x/(w-1.0) - 0.5)
    Y = 2*(grid_y/(h-1.0) - 0.5)


    grid_tf = torch.stack((X,Y), dim=3)

    flow_sample = torch.nn.functional.grid_sample(flow, grid_tf, padding_mode=padding_mode)

    return flow_sample

def compute_triangulation_loss(pose,flow,intrinsics, depth_pre): # intrinsic scale
    flow = grid_sample(flow[0],8)
    B, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]
    translation_x = pose[:, 0] # [B 1]

    zeros = translation_x.detach()*0
    ones = zeros.detach()+1

    projection_matrix_A = torch.stack([ones, zeros, zeros, zeros,
                        zeros, ones, zeros, zeros,
                        zeros, zeros,  ones, zeros], dim=1).view(B, 3, 4)   
    projection_matrix_B = pose_vec2mat(pose)

    # projection_matrix_A = (intrinsics.bmm(projection_matrix_A)).unsqueeze(1).expand(B,h*w,3,4)
    # projection_matrix_B = (intrinsics.bmm(projection_matrix_B)).unsqueeze(1).expand(B,h*w,3,4)

    projection_matrix_A = torch.bmm(intrinsics,projection_matrix_A).unsqueeze(1).repeat(1,h*w,1,1)
    projection_matrix_B = torch.bmm(intrinsics,projection_matrix_B).unsqueeze(1).repeat(1,h*w,1,1)

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]
    
    X = grid_x + u # [B h w]
    Y = grid_y + v # [B h w]

    uA = grid_x.view(B, h*w, 1, 1)
    vA = grid_y.view(B, h*w, 1, 1)


    uB = X.view(B, h*w, 1, 1)
    vB = Y.view(B, h*w, 1, 1)

    depth = []
    
    for batch_i in range(B):
        projection_matrix_A_batch = projection_matrix_A[batch_i,:,:,:]
        projection_matrix_B_batch = projection_matrix_B[batch_i,:,:,:]
        uA_batch = uA[batch_i,:,:,:] # N 1 1
        vA_batch = vA[batch_i,:,:,:]
        uB_batch = uB[batch_i,:,:,:]
        vB_batch = vB[batch_i,:,:,:]
        PA1_batch = projection_matrix_A_batch[:,0,:].unsqueeze(1) # N 1 4
        PA2_batch = projection_matrix_A_batch[:,1,:].unsqueeze(1)
        PA3_batch = projection_matrix_A_batch[:,2,:].unsqueeze(1)
        PB1_batch = projection_matrix_B_batch[:,0,:].unsqueeze(1)
        PB2_batch = projection_matrix_B_batch[:,1,:].unsqueeze(1)
        PB3_batch = projection_matrix_B_batch[:,2,:].unsqueeze(1)

        row1_batch = torch.mul(uA_batch,PA3_batch)-PA1_batch
        row2_batch = torch.mul(vA_batch,PA3_batch)-PA2_batch
        row3_batch = torch.mul(uB_batch,PB3_batch)-PB1_batch
        row4_batch = torch.mul(vB_batch,PB3_batch)-PB2_batch

        A = torch.cat((row1_batch,row2_batch,row3_batch,row4_batch),dim=1)
        u, _, v = torch.svd(A)
        depth_batch = v[:,2,3]/v[:,3,3]
        depth_batch = depth_batch.view(h,w)
        depth_batch = depth_batch.unsqueeze(0).unsqueeze(0)
        depth.append(depth_batch)
    
    depth = torch.cat(depth,0)
    depth_mean = depth.mean()
    depth_pre_mean = depth_pre[0].mean()
    scale = depth_pre_mean/depth_mean
    depth *= scale
    depth_pre = Variable( grid_sample(depth_pre[0],8).detach().data, requires_grad=False) ##edited
    loss = robust_l1(depth-depth_pre)

    return loss, depth
