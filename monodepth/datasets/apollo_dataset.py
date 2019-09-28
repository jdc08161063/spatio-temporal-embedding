# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil

from glob import glob
from tqdm import tqdm

from monodepth.datasets.mono_dataset import MonoDataset


class ApolloDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mode = 'train' if self.is_train else 'val'

        # Defined as in apollo/utilities/intrinsics.txt
        # fx, 0, Cx / 3384 and fy, 0, Cy / 2710
        self.K = np.array([[0.68101, 0, 0.49830, 0],
                           [0, 0.85088, 0.49999, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (256, 128)  # (width, height)

    def check_depth(self):
        # Do not load ground truth depth map
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:04d}_image{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, self.mode, folder, f_str)
        return image_path


def generate_split():
    root = ''
    side = 'l'

    for mode in ['train', 'val']:
        output_file = ''.format(mode)
        scene_names = glob(os.path.join(root, mode, '*'))
        for scene in tqdm(scene_names, total=len(scene_names)):
            img_filenames = sorted(glob(os.path.join(scene, '*.jpg')))
            assert '{:04d}'.format(len(img_filenames) - 1) == os.path.basename(img_filenames[-1])[:-len('_image.jpg')]
            # Remove first and last element
            img_filenames = img_filenames[1:-1]
            base_folder = os.path.basename(scene)

            with open(output_file, 'a') as f:
                for t in range(len(img_filenames)):
                    f.write('{} {} {}\n'.format(base_folder, t + 1, side))


if __name__ == '__main__':
    generate_split()
