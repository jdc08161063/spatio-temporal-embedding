import os

import numpy as np

from glob import glob
from PIL import Image
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from data.utils import crop_and_resize, pil_loader


def preprocess_dataset(root='', root_save='',
                       img_size=(192, 640), keep_id=1, extension='.png'):
    pool = Pool(cpu_count() - 1)
    for mode in ['train', 'val']:
        all_scene_dir = sorted(glob(os.path.join(root, 'images', mode, '*')))
        for _ in tqdm(pool.imap_unordered(
                partial(preprocess_dataset_iter, root=root, mode=mode, root_save=root_save, img_size=img_size,
                        keep_id=keep_id, extension=extension), all_scene_dir), total=len(all_scene_dir)):
            pass


def preprocess_dataset_iter(scene_dir, root, mode, root_save, img_size=(192, 640), keep_id=1, extension='.png'):
    """ keep_id 1 is car, 2 is pedestrian"""
    print('Scene: {}'.format(scene_dir))
    img_filenames = sorted(glob(os.path.join(root, 'images', mode, os.path.basename(scene_dir), '*' + extension)))

    for img_fname in img_filenames:
        basename = os.path.basename(img_fname)
        img = pil_loader(img_fname)  # Open and convert to RGB
        #  Much more compact to load and resize images compared to numpy arrays
        instance_seg = Image.open(os.path.join(root, 'instances', mode, os.path.basename(scene_dir),
                                               basename[:-len(extension)] + '.png'))
        assert instance_seg.mode == 'I'
        #####
        #  Resize
        img = crop_and_resize(img, img_size, crop=False)
        instance_seg = crop_and_resize(instance_seg, img_size, order=0, crop=False)

        # Filter non-cars
        instance_seg = np.array(instance_seg)
        unique_ids = np.unique(instance_seg)
        # Only keep cars, remove 10000 (ignore regions) and 200x (pedestrians)
        keep_ids = [id for id in unique_ids if (str(id).startswith(str(keep_id)) and len(str(id)) == 4)]
        instance_seg[~np.isin(instance_seg, keep_ids)] = 0
        instance_seg = (instance_seg % 1000).astype(np.uint8)

        #  N_CLASSES = 1
        instance_seg = instance_seg[None, :, :]

        save_path = os.path.join(root_save, mode, os.path.basename(scene_dir))
        os.makedirs(save_path, exist_ok=True)
        img.save(os.path.join(save_path, basename[:-len(extension)] + '_image.jpg'))
        np.save(os.path.join(save_path, basename[:-len(extension)] + '_instance_seg.npy'), instance_seg)
