from __future__ import division
import json
import numpy as np
import scipy.misc
from path import Path
from tqdm import tqdm


class sintel_loader(object):
    def __init__(self,
                 dataset_dir,
                 split='training',
                 img_height=256,
                 img_width=832):
        self.dataset_dir = Path(dataset_dir)
        self.split = split

        self.img_height = img_height
        self.img_width = img_width

        self.scenes = (self.dataset_dir/split/'final').dirs()
        print('Total scenes collected: {}'.format(len(self.scenes)))

        
    def get_scene_imgs(self, scene):
        img_files = sorted(scene.files('*.png'))
        for img_file in img_files:
            yield self.load_image(img_file)


    def load_image(self, img_path):
        if not img_path.isfile():
            print("invalid path")
            return None
        img = scipy.misc.imread(img_path)
        img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img
