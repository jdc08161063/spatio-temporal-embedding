import os
import shutil

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from PIL import Image
from functools import partial
from multiprocessing import Pool, cpu_count

from common.constants import MAX_INSTANCES, MAX_INSTANCES_SCENE
from data.consistency import enforce_consistency, make_ids_consecutives
from data.utils import crop_and_resize

#########################
### Dataset creation
# 1. Create apollo.csv with `create_apollo_df`
# 2. Call `split_dataset` to split into train/val, containing folders {road_nb}_{record}_{camera_nb}
#    with img (.jpg), instance_seg (.png), semantic_seg (.png)
# 3. Call `preprocess_dataset` to make instance ids consistent and resize to 128x256.
#########################

ROOT = ''
CSV_FILENAME = 'apollo.csv'
CAMERA = 5
IMG_SIZE = (128, 256)
LABEL_ID = 33  # ids of car
MIN_PIXEL_INSTANCE = 100


def split_dataset(root_save=''):
    pool = Pool(cpu_count() - 1)
    df_apollo = pd.read_csv(os.path.join(ROOT, CSV_FILENAME))

    for mode in ['train', 'val']:
        unique_road_nb = np.unique(df_apollo[df_apollo['split'] == mode]['road_nb'])

        for road_nb in unique_road_nb:  # TODO remove
            df_apollo_road_nb = df_apollo[(df_apollo['road_nb'] == road_nb) & (df_apollo['split'] == mode)]
            unique_records = np.unique(df_apollo_road_nb['record'])

            for _ in tqdm(pool.imap_unordered(
                    partial(split_dataset_iter, df_apollo_road_nb=df_apollo_road_nb, root_save=root_save, mode=mode,
                            road_nb=road_nb), unique_records), total=len(unique_records)):
                pass


def split_dataset_iter(record, df_apollo_road_nb, root_save, mode, road_nb):
    path = os.path.join(root_save, mode, road_nb + '_' + record + '_' + str(CAMERA))
    os.makedirs(path, exist_ok=True)
    filter_mask = (df_apollo_road_nb['record'] == record) & (df_apollo_road_nb['camera'] == CAMERA)
    tuple_filenames = zip(df_apollo_road_nb[filter_mask]['img_path'],
                          df_apollo_road_nb[filter_mask]['sem_path'],
                          df_apollo_road_nb[filter_mask]['inst_path']
                          )
    print('Road: {}, {}'.format(road_nb, record))
    count = 0
    for f_img, f_semantic, f_instance_seg in tuple_filenames:
        shutil.copy(src=os.path.join(ROOT, f_img),
                    dst=os.path.join(path, '{:04d}_image_tmp.jpg'.format(count)))  # TODO: remove these tmp

        shutil.copy(src=os.path.join(ROOT, f_semantic),
                    dst=os.path.join(path, '{:04d}_semantic_tmp.png'.format(count)))

        shutil.copy(src=os.path.join(ROOT, f_instance_seg),
                    dst=os.path.join(path, '{:04d}_instance_seg_tmp.png'.format(count)))

        count += 1


def preprocess_dataset(root='',
                       root_save=''):
    pool = Pool(cpu_count() - 1)
    for mode in ['train', 'val']:
        all_scene_dir = sorted(glob(os.path.join(root, mode, '*')))
        for _ in tqdm(pool.imap_unordered(
                partial(preprocess_dataset_iter, root=root, mode=mode, root_save=root_save), all_scene_dir),
                total=len(all_scene_dir)):
            pass


def preprocess_dataset_iter(scene_dir, root, mode, root_save):
    print('Scene: {}'.format(scene_dir))
    img_filenames = sorted(glob(os.path.join(root, mode, os.path.basename(scene_dir), '*_image_tmp.jpg')))

    available_keys = set(range(1, MAX_INSTANCES_SCENE))  # max of 256 instances
    dict_ids = None
    prev_instance_seg = None
    for img_fname in img_filenames:
        basename = img_fname[:-len('image_tmp.jpg')]
        img = Image.open(img_fname)
        #  Much more compact to load and resize images compared to numpy arrays
        semantic = Image.open(basename + 'semantic_tmp.png')
        instance_seg = Image.open(basename + 'instance_seg_tmp.png')

        #####
        #  Resize
        img = crop_and_resize(img, IMG_SIZE)
        instance_seg = crop_and_resize(instance_seg, IMG_SIZE, order=0)
        semantic = crop_and_resize(semantic, IMG_SIZE, order=0)
        # depth = crop_and_resize(depth, IMG_SIZE, order=1)

        #####
        # Filter instance ids
        instance_seg = np.array(instance_seg)
        unique_ids = np.unique(instance_seg)
        # The relevant ids start with '33'
        relevant_ids = [x for x in unique_ids if str(x).startswith(str(LABEL_ID))][:MAX_INSTANCES]
        # Remove too small instances
        relevant_ids = [x for x in relevant_ids if np.sum(instance_seg == x) > MIN_PIXEL_INSTANCE]
        mask = np.isin(instance_seg, relevant_ids)
        instance_seg[~mask] = 0

        instance_seg = make_ids_consecutives(instance_seg)

        #  N_CLASSES = 1
        instance_seg = instance_seg[None, :, :]

        ######
        # Enforce consistency
        consistent_instance_seg, available_keys, dict_ids = enforce_consistency(instance_seg, prev_instance_seg,
                                                                                available_keys, dict_ids)

        #  The algorithm only works when for each frame, instance ids are in [0, max_n_instance[
        prev_instance_seg = instance_seg

        ######
        #  Save
        save_path = os.path.join(root_save, mode, os.path.basename(scene_dir))
        os.makedirs(save_path, exist_ok=True)
        img.save(os.path.join(save_path, os.path.basename(basename) + 'image.jpg'))
        np.save(os.path.join(save_path, os.path.basename(basename) + 'instance_seg.npy'), consistent_instance_seg)
        np.save(os.path.join(save_path, os.path.basename(basename) + 'semantic.npy'), np.array(semantic))


def create_apollo_df():
    """ Create initial Apollo dataframe containing the train/val split as defined by ApolloScape. (only do it once)

    Parameters
    ----------
        ROOT: directory containing the ApolloScape dataset
        CSV_FILENAME: filename of the saved dataframe as a csv file
    """
    apollo_df = pd.DataFrame(columns=['split', 'road_nb', 'record', 'camera',
                                      'img_path', 'sem_path', 'inst_path'])

    for split in ['train', 'val']:
        for road_nb in ['road01_ins', 'road02_ins', 'road03_ins']:
            print(road_nb)
            split_df = pd.read_csv(os.path.join(ROOT, 'dataset_splits', road_nb + '_' + split + '.lst'), sep='\t',
                                   header=None, names=['image_path', 'semantic_path'])
            print(split_df.shape)

            # All the image and label filenames (relative path)
            im_filenames = glob(os.path.join(ROOT, road_nb, 'ColorImage/*/*/*'))
            im_filenames = set([os.path.relpath(x, ROOT) for x in im_filenames])
            label_filenames = glob(os.path.join(ROOT, road_nb, 'Label/*/*/*'))
            label_filenames = set([os.path.relpath(x, ROOT) for x in label_filenames])

            inst_not_found = 0
            for im, sem in split_df[['image_path', 'semantic_path']].values:
                # Check that the files exist
                assert im in im_filenames, 'Image not found'
                assert sem in label_filenames, 'Semantic seg. not found'

                # Extract record, camera, image id
                _, _, record, camera, im_id = os.path.normpath(im).split(os.sep)
                camera_nb = camera[-1]
                im_id = im_id.split('.jpg')[0]  # gives something like '170908_085500604_Camera_6'

                # Deduce instance ids filename
                inst = os.path.join(os.path.dirname(sem), im_id + '_instanceIds.png')
                if inst not in label_filenames:
                    inst_not_found += 1

                apollo_df = apollo_df.append(
                    pd.DataFrame({'split': split, 'road_nb': road_nb, 'record': record, 'camera': camera_nb,
                                  'img_path': im, 'sem_path': sem, 'inst_path': inst}, index=[0]),
                    sort=False)

            print('Instance not found: ', inst_not_found)

    apollo_df = apollo_df.reset_index(drop=True)
    apollo_df.to_csv(os.path.join(ROOT, CSV_FILENAME), index=False)
    print('Finished! Saved as csv file')
