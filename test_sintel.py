# Author: Anurag Ranjan
# Copyright (c) 2018, Max Planck Society 
# python test_flow_back2future.py --flownet FlowNetC6  --pretrained-flow '/home/ljf/triangulation/checkpoints/flow_0112_b16_pf1st0_pf2nd3_s1st0_s2nd10_alpha10_cv0_ssim0_min/flownet_model_best.pth.tar'    --kitti-dir '/home/ljf/FullGeoNet/kitti/kitti2015' 

import argparse
import torch
import torch.nn as nn
from path import Path
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from scipy.misc import imread, imresize
from tqdm import tqdm
import numpy as np
import models
import custom_transforms
from flowutils import flow_io
from sintel_eval import sintel_io

parser = argparse.ArgumentParser(description='Code to test performace of Back2Future models on KITTI benchmarks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--flownet', dest='flownet', type=str, default='Back2Future', choices=['Back2Future','Back2FutureSN','Back2FutureS','FlowNetC6'],
                    help='flow network architecture. Options: FlowNetS | SpyNet')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                    help='path to pre-trained Flow net model')
parser.add_argument('--sintel-dir', dest='sintel_dir', default=None, metavar='PATH',
                    help='path to Sintel directory')
parser.add_argument('--dsplit', dest='dsplit', type=str, default='final', choices=['final','clean'],
                    help='which dataset final or clean')
                 
def main():
    global args
    normalize = Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    args = parser.parse_args()
    flow_loader_h, flow_loader_w = 256, 832
    # flow_loader_h, flow_loader_w = 384, 1024
    # valid_flow_transform = Scale(h=flow_loader_h, w=flow_loader_w)

    valid_flow_transform = Compose([Scale(h=flow_loader_h, w=flow_loader_w),
                            ArrayToTensor(), normalize])

    val_flow_set = Sintel(root=args.sintel_dir, split=args.dsplit,transform=valid_flow_transform)

    val_flow_loader = torch.utils.data.DataLoader(val_flow_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True)

    flow_net = getattr(models, args.flownet)(nlevels=6).cuda()
    if args.pretrained_flow:
        print("=> using pre-trained weights from {}".format(args.pretrained_flow))
        weights = torch.load(args.pretrained_flow)
        flow_net.load_state_dict(weights['state_dict'])#, strict=False)
    flow_net.eval()
    error_names = ['epe_total']
    errors = AverageMeter(i=len(error_names))


    for i, (tgt_img, ref_img, flow_gt) in enumerate(tqdm(val_flow_loader)):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        ref_img_var = Variable(ref_img.cuda(), volatile=True)
        flow_gt_var = Variable(flow_gt.cuda(), volatile=True)

        # compute output
        flow_fwd = flow_net(tgt_img_var, ref_img_var)
        epe = compute_epe(gt=flow_gt_var, pred=flow_fwd[0].unsqueeze(0))
        errors.update(epe)

    print("Averge EPE",errors.avg )

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images

class Scale(object):
    """Scales images to a particular size"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images):

        in_h, in_w, _ = images[0].shape
        scaled_h, scaled_w = self.h , self.w
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        return scaled_images


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors


def crawl_folders_seq(folders_list):
        imgs1 = []
        imgs2 = []
        flow = []
        for folder in folders_list:
            current_imgs = sorted(folder.files('*.png'))
            current_imgs1 = current_imgs[:-1]
            current_imgs2 = current_imgs[1:]
            current_flow = []
            for (img1,img2) in zip(current_imgs1, current_imgs2):
                scene_name = img1.dirname().split('/')[-1]
                dirpath = img1.dirname().dirname().dirname()
                f = dirpath/'flow'/scene_name/(img1.name[:-4] + '.flo')
                assert(f.isfile()), "flow file {} not found".format(str(f))
                current_flow.append(f)
            imgs1.extend(current_imgs1)
            imgs2.extend(current_imgs2)
            flow.extend(current_flow)
        return imgs1, imgs2, flow


class Sintel(torch.utils.data.Dataset):
    """
        Sintel loader
    """
    def __init__(self, root, split='final', transform=None):
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/'training'/split/folder[:-1] for folder in open(scene_list_path)]
        self.imgs1, self.imgs2, self.gtFlows = crawl_folders_seq(self.scenes)
        self.transform = transform

    def __getitem__(self, index):
        img1 = load_as_float(self.imgs1[index])
        img2 = load_as_float(self.imgs2[index])
        u,v = sintel_io.flow_read(self.gtFlows[index])
        gtFlow = np.dstack((u,v))
        gtFlow = torch.FloatTensor(gtFlow.transpose(2,0,1))
        if self.transform is not None:
            img = self.transform([img1, img2])
            img1, img2 = img[0], img[1]

        return img1, img2, gtFlow

    def __len__(self):
        return len(self.imgs1)


# class Scale(object):
#     """Scales images to a particular size"""
#     def __init__(self, h, w):
#         self.h = h
#         self.w = w

#     def __call__(self, images):
#         in_h, in_w, _ = images[0].shape
#         scaled_h, scaled_w = self.h , self.w
#         scaled_images = [ToTensor()(imresize(im, (scaled_h, scaled_w))) for im in images]
#         return scaled_images

def compute_epe(gt, pred):
    # print(pred.size())
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
        avg_epe = epe.sum()/(valid.sum() + 1e-6)
    else:
        avg_epe = epe.sum()/(bs*h_gt*w_gt)

    if type(avg_epe) == Variable: avg_epe = avg_epe.data
    return avg_epe.item()

def compute_all_epe(gt_occ, gt_noc, pred):
    # print(pred.size())
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt_noc.size()

    u_occ_gt, v_occ_gt = gt_occ[:,0,:,:], gt_occ[:,1,:,:]
    u_noc_gt, v_noc_gt = gt_noc[:,0,:,:], gt_noc[:,1,:,:]

    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe_occ = torch.sqrt(torch.pow((u_occ_gt - u_pred), 2) + torch.pow((v_occ_gt - v_pred), 2))
    epe_noc = torch.sqrt(torch.pow((u_noc_gt - u_pred), 2) + torch.pow((v_noc_gt - v_pred), 2))

    valid_occ = gt_occ[:,2,:,:]
    valid_noc = gt_noc[:,2,:,:]

    epe_occ = epe_occ * valid_occ
    epe_noc = epe_noc * valid_noc

    avg_epe = (epe_occ.sum()+epe_noc.sum())/(valid_occ.sum()+valid_noc.sum() + 1e-6)

    if type(avg_epe) == Variable: avg_epe = avg_epe.data
    return avg_epe.item()

def load_as_float(path):
    return imread(path).astype(np.float32)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)

if __name__ == '__main__':
    main()
