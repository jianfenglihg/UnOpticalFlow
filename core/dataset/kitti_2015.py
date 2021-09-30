import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kitti_2012 import KITTI_2012

class KITTI_2015(KITTI_2012):
    def __init__(self, data_dir, img_hw=(256, 832)):
        super(KITTI_2015, self).__init__(data_dir, img_hw, init=False)
        self.num_total = 200

        self.data_list = self.get_data_list()

    def get_data_list(self):
        data_list = []
        for i in range(self.num_total):
            data = {}
            data['img1_dir'] = os.path.join(self.data_dir, 'image_2', str(i).zfill(6) + '_10.png')
            data['img2_dir'] = os.path.join(self.data_dir, 'image_2', str(i).zfill(6) + '_11.png')
            data['calib_file_dir'] = os.path.join(self.data_dir, 'calib_cam_to_cam', str(i).zfill(6) + '.txt')
            data_list.append(data)
        return data_list        

if __name__ == '__main__':
    pass

