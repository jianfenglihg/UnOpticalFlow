from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize, imrotate

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics


class NormalizeLocally(object):

    def __call__(self, images, intrinsics):
        image_tensor = torch.stack(images)
        assert(image_tensor.size(1)==3)   #3 channel image
        mean = image_tensor.transpose(0,1).contiguous().view(3, -1).mean(1)
        std = image_tensor.transpose(0,1).contiguous().view(3, -1).std(1)

        for tensor in images:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0,2] = w - output_intrinsics[0,2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomVerticalFlip(object):
    """Randomly vertically flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() > 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.flipud(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0,2] = w - output_intrinsics[0,2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics

class RandomRotate(object):
    """Randomly rotates images up to 10 degrees and crop them to keep same size as before."""
    def __call__(self, images, intrinsics):
        if np.random.random() > 0.5:
            return images, intrinsics
        else:
            assert intrinsics is not None
            rot = np.random.uniform(0,10)
            rotated_images = [imrotate(im, rot) for im in images]

            return rotated_images, intrinsics

# def random_channel_swap(img_list):
#     channel_permutation = tf.constant([[0, 1, 2],
#                                        [0, 2, 1],
#                                        [1, 0, 2],
#                                        [1, 2, 0], 
#                                        [2, 0, 1],
#                                        [2, 1, 0]])    
#     rand_i = tf.random_uniform([], minval=0, maxval=6, dtype=tf.int32)
#     perm = channel_permutation[rand_i]
#     for i, img in enumerate(img_list):
#         channel_1 = img[:, :, perm[0]]
#         channel_2 = img[:, :, perm[1]]
#         channel_3 = img[:, :, perm[2]]
#         img_list[i] = tf.stack([channel_1, channel_2, channel_3], axis=-1)
#     return img_list



# def augmentation(self, img1, img2):
#     img1, img2 = random_crop([img1, img2], self.crop_h, self.crop_w)
#     img1, img2 = random_flip([img1, img2])
#     img1, img2 = random_channel_swap([img1, img2])
#     return img1, img2 


class Histogram_equalization(object):
    def rgb_histeql(im):
        imarr = im.flatten()
        hist, bins = np.histogram(imarr, 255)
        cdf = np.cumsum(hist)
        cdf = 255* (cdf/cdf[-1])
        imarr2 = np.interp(imarr, bins[:-1], cdf)
        imarr2 = imarr2.reshape(im.shape)
        return imarr2

    def __call__(self, images, intrinsics):
        histeql_images = [rgb_histeql(im) for im in images]
        return histeql_images, intrinsics

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""
    def __init__(self, h=0, w=0):
        self.h = h
        self.w = w

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.1,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        if self.h and self.w:
            in_h, in_w = self.h, self.w

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, output_intrinsics

class Scale(object):
    """Scales images to a particular size"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        scaled_h, scaled_w = self.h , self.w

        output_intrinsics[0] *= (scaled_w / in_w)
        output_intrinsics[1] *= (scaled_h / in_h)
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        return scaled_images, output_intrinsics
