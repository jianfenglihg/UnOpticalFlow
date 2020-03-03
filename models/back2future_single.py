# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from spatial_correlation_sampler import spatial_correlation_sample

def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=9,
                                          stride=1)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return out_corr

def conv_feat_block(nIn, nOut):
    return nn.Sequential(
        nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(nOut, nOut, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2)
    )

def conv_dec_block(nIn):
    return nn.Sequential(
        nn.Conv2d(nIn, 128, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
    )


class Model(nn.Module):
    def __init__(self, nlevels):
        super(Model, self).__init__()

        self.nlevels = nlevels
        idx = [list(range(n, -1, -9)) for n in range(80,71,-1)]
        idx = list(np.array(idx).flatten())
        self.idx_fwd = Variable(torch.LongTensor(np.array(idx)).cuda(), requires_grad=False)
        self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear')
        self.softmax2d = nn.Softmax2d()

        self.conv1a = conv_feat_block(3,16)
        self.conv1b = conv_feat_block(3,16)


        self.conv2a = conv_feat_block(16,32)
        self.conv2b = conv_feat_block(16,32)


        self.conv3a = conv_feat_block(32,64)
        self.conv3b = conv_feat_block(32,64)


        self.conv4a = conv_feat_block(64,96)
        self.conv4b = conv_feat_block(64,96)


        self.conv5a = conv_feat_block(96,128)
        self.conv5b = conv_feat_block(96,128)


        self.conv6a = conv_feat_block(128,192)
        self.conv6b = conv_feat_block(128,192)


        self.corr = correlate # Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)

        self.decoder_fwd6 = conv_dec_block(81)
       
        self.decoder_fwd5 = conv_dec_block(211)
       
        self.decoder_fwd4 = conv_dec_block(179)
       
        self.decoder_fwd3 = conv_dec_block(147)
      
        self.decoder_fwd2 = conv_dec_block(115)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def normalize(self, ims):
        imt = []
        for im in ims:
            im = im * 0.5
            im = im + 0.5
            im[:,0,:,:] = im[:,0,:,:] - 0.485  # Red
            im[:,1,:,:] = im[:,1,:,:] - 0.456 # Green
            im[:,2,:,:] = im[:,2,:,:] - 0.406 # Blue

            im[:,0,:,:] = im[:,0,:,:] / 0.229  # Red
            im[:,1,:,:] = im[:,1,:,:] / 0.224 # Green
            im[:,2,:,:] = im[:,2,:,:] / 0.225 # Blue

            imt.append(im)
        return imt

    def normalize_tgt(self, ims):
        # tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - 0.411618
		# tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - 0.434631
		# tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - 0.454253

		# tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - 0.410782
		# tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - 0.433645
		# tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - 0.452793
        imt = []
        for im in ims:
            im[:,0,:,:] = im[:,0,:,:] - 0.411618  # Red
            im[:,1,:,:] = im[:,1,:,:] - 0.434631 # Green
            im[:,2,:,:] = im[:,2,:,:] - 0.454253 # Blue

            imt.append(im)
        return imt

    def normalize_ref(self, ims):
        # tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - 0.411618
		# tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - 0.434631
		# tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - 0.454253

		# tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - 0.410782
		# tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - 0.433645
		# tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - 0.452793
        imt = []
        for im in ims:
            im[:,0,:,:] = im[:,0,:,:] - 0.410782 # Red
            im[:,1,:,:] = im[:,1,:,:] - 0.433645 # Green
            im[:,2,:,:] = im[:,2,:,:] - 0.452793 # Blue

            imt.append(im)
        return imt


    def forward(self, im_tar, im_ref):
        '''
            inputS:
                im_tar: Middle Frame, I_0
                im_ref: Adjecent Frames in the order

            outputs:
                At self.nlevels different scales:
                flow_fwd: optical flow from I_0 to I+

        '''
        # im = Variable(torch.zeros(1,9,512,512).cuda())
        # ima = im[:, :3, :, :] + 0.2     # I_0
        # imb = im[:, 3:6, :, :] + 0.3    # I_+
        # imc = im[:, 6:, :, :] + 0.1     # I_-
        
        # im_norm = self.normalize([im_tar] + [im_ref])

        im_norm_ref = self.normalize_ref([im_ref])
        im_norm_tgt = self.normalize_tgt([im_tar])
        im_norm = im_norm_tgt+im_norm_ref

        # feat1a = self.conv1a(im_tar)
        feat1a = self.conv1a(im_norm[0])
        feat2a = self.conv2a(feat1a)
        feat3a = self.conv3a(feat2a)
        feat4a = self.conv4a(feat3a)
        feat5a = self.conv5a(feat4a)
        feat6a = self.conv6a(feat5a)

        # feat1b = self.conv1b(im_ref)
        feat1b = self.conv1b(im_norm[1])
        feat2b = self.conv2b(feat1b)
        feat3b = self.conv3b(feat2b)
        feat4b = self.conv4b(feat3b)
        feat5b = self.conv5b(feat4b)
        feat6b = self.conv6b(feat5b)

        corr6_fwd = self.corr(feat6a, feat6b)
        corr6_fwd = corr6_fwd.index_select(1,self.idx_fwd)

        flow6_fwd = self.decoder_fwd6(corr6_fwd)
        flow6_fwd_up = self.upsample(flow6_fwd)
        feat5b_warped = self.warp(feat5b, 0.625*flow6_fwd_up)


        corr5_fwd = self.corr(feat5a, feat5b_warped)
        corr5_fwd = corr5_fwd.index_select(1,self.idx_fwd)

        upfeat5_fwd = torch.cat((corr5_fwd, feat5a, flow6_fwd_up), 1)
        flow5_fwd = self.decoder_fwd5(upfeat5_fwd)
        flow5_fwd_up = self.upsample(flow5_fwd)
        feat4b_warped = self.warp(feat4b, 1.25*flow5_fwd_up)

        corr4_fwd = self.corr(feat4a, feat4b_warped)
        corr4_fwd = corr4_fwd.index_select(1,self.idx_fwd)

        upfeat4_fwd = torch.cat((corr4_fwd, feat4a, flow5_fwd_up), 1)
        flow4_fwd = self.decoder_fwd4(upfeat4_fwd)
        flow4_fwd_up = self.upsample(flow4_fwd)
        feat3b_warped = self.warp(feat3b, 2.5*flow4_fwd_up)
        

        corr3_fwd = self.corr(feat3a, feat3b_warped)
        corr3_fwd = corr3_fwd.index_select(1,self.idx_fwd)


        upfeat3_fwd = torch.cat((corr3_fwd, feat3a, flow4_fwd_up), 1)
        flow3_fwd = self.decoder_fwd3(upfeat3_fwd)
        flow3_fwd_up = self.upsample(flow3_fwd)
        feat2b_warped = self.warp(feat2b, 5.0*flow3_fwd_up)
 

        corr2_fwd = self.corr(feat2a, feat2b_warped)
        corr2_fwd = corr2_fwd.index_select(1,self.idx_fwd)


        upfeat2_fwd = torch.cat((corr2_fwd, feat2a, flow3_fwd_up), 1)
        flow2_fwd = self.decoder_fwd2(upfeat2_fwd)
        flow2_fwd_up = self.upsample(flow2_fwd)
       

        flow2_fwd_fullres = 20*self.upsample(flow2_fwd_up)
        flow3_fwd_fullres = 10*self.upsample(flow3_fwd_up)
        flow4_fwd_fullres = 5*self.upsample(flow4_fwd_up)
        flow5_fwd_fullres = 2.5*self.upsample(flow5_fwd_up)
        flow6_fwd_fullres = 1.25*self.upsample(flow6_fwd_up)


        if self.training:
            flow_fwd = [flow2_fwd_fullres, flow3_fwd_fullres, flow4_fwd_fullres, flow5_fwd_fullres, flow6_fwd_fullres]

            if self.nlevels==6:
                flow_fwd.append(0.625*flow6_fwd_up)
                
            return flow_fwd
        else:
            return flow2_fwd_fullres


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
        mask = torch.autograd.Variable(torch.ones(x.size()), requires_grad=False).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask.data<0.9999] = 0
        mask[mask.data>0] = 1

        return output#*mask
