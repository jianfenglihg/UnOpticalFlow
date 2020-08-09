import os, sys
import numpy as np
import cv2
from tqdm import tqdm
import torch.multiprocessing as mp
import pdb

def process_folder(q, data_dir, output_dir, stride=1):
    while True:
        if q.empty():
            break
        folder = q.get()
        image_path = os.path.join(data_dir, folder)
        dump_image_path = os.path.join(output_dir, folder)
        if not os.path.isdir(dump_image_path):
            os.makedirs(dump_image_path)
        f = open(os.path.join(dump_image_path, 'train.txt'), 'w')
        
        # Note. the os.listdir method returns arbitary order of list. We need correct order.
        numbers = len(os.listdir(image_path))
        names = list(os.listdir(image_path))
        names.sort()
        if numbers < 3:
            print("this folder do not have enough image, numbers < 3!")
        for n in range(numbers - 2*stride):
            s_idx = n
            m_idx = s_idx + stride
            e_idx = s_idx + 2*stride
            
            #curr_image = cv2.imread(os.path.join(image_path, '%.5d'%s_idx)+'.png')
            #middle_image = cv2.imread(os.path.join(image_path, '%.5d'%m_idx)+'.png')
            #next_image = cv2.imread(os.path.join(image_path, '%.5d'%e_idx)+'.png')
            curr_image = cv2.imread(os.path.join(image_path, names[s_idx]))
            middle_image = cv2.imread(os.path.join(image_path, names[m_idx]))
            next_image = cv2.imread(os.path.join(image_path, names[e_idx]))

            if curr_image is None:
                print(os.path.join(image_path, '%.5d'%s_idx)+'.png')
                continue

            if middle_image is None:
                print(os.path.join(image_path, '%.5d'%m_idx)+'.png')
                continue

            if next_image is None:
                print(os.path.join(image_path, '%.5d'%e_idx)+'.png')
                continue

            seq_images = np.concatenate([curr_image, middle_image, next_image], axis=0)
            cv2.imwrite(os.path.join(dump_image_path, '%.10d'%s_idx)+'.png', seq_images.astype('uint8'))

            # Write training files
            f.write('%s\n' % (os.path.join(folder, '%.10d'%s_idx)+'.png'))
        print(folder)


class SINTEL(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        raise NotImplementedError


    def prepare_data_mp(self, output_dir, stride=1):
        num_processes = 8
        processes = []
        q = mp.Queue()
        if not os.path.isfile(os.path.join(output_dir, 'train.txt')):
            os.makedirs(output_dir)
            #f = open(os.path.join(output_dir, 'train.txt'), 'w')
            print('Preparing sequence data....')
            if not os.path.isdir(self.data_dir):
                raise NotImplementedError
            dirlist = os.listdir(self.data_dir)
            total_dirlist = []
            # Get the different folders of images
            for d in dirlist:
                if os.path.isdir(os.path.join(self.data_dir, d)):
                    total_dirlist.append(d)
                    q.put(d)
            # Process every folder
            for rank in range(num_processes):
                p = mp.Process(target=process_folder, args=(q, self.data_dir, output_dir, stride))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        
        # Collect the training frames.
        f = open(os.path.join(output_dir, 'train.txt'), 'w')
        for date in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, date)):
                train_file = open(os.path.join(output_dir, date, 'train.txt'), 'r')
                for l in train_file.readlines():
                    f.write(l)
        
       
        print('Data Preparation Finished.')


    def __getitem__(self, idx):
        raise NotImplementedError


if __name__ == '__main__':
    data_dir = '/home/ljf/Dataset/Sintel/scene'
    dirlist = os.listdir('/home4/zhaow/data/kitti')
    output_dir = '/home4/zhaow/data/kitti_seq/data_generated_s2'
    total_dirlist = []
    # Get the different folders of images
    for d in dirlist:
        seclist = os.listdir(os.path.join(data_dir, d))
        for s in seclist:
            if os.path.isdir(os.path.join(data_dir, d, s)):
                total_dirlist.append(os.path.join(d, s))
    
    F = open(os.path.join(output_dir, 'train.txt'), 'w')
    for p in total_dirlist:
        traintxt = os.path.join(os.path.join(output_dir, p), 'train.txt')
        f = open(traintxt, 'r')
        for line in f.readlines():
            F.write(line)
        print(traintxt)





