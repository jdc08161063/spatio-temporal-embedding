import os
import torch

import numpy as np
import torchvision.transforms as transforms

from skimage.transform import resize
from PIL import Image
from glob import glob
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from torch.utils.data import Dataset

from common.constants import MAX_INSTANCES, N_CLASSES, MEAN, STD
from data.utils import pil_loader


class MotionDataset(Dataset):
    def __init__(self, root, dataset='nuscenes', mode='', seq_len=5, h=128, w=256, load_depth_inputs=False,
                 num_scales=3, saved_numpy=False):
        assert dataset in ['nuscenes', 'apollo', 'kitti', 'kitti_ped', 'davis'], 'Not recognised dataset.'
        assert seq_len >= 3, 'Sequence length={} but must be greater of equal than 3.'.format(seq_len)

        self.dataset = dataset
        self.seq_len = seq_len
        # Original image in nuscenes is 900 x 1600. Divide by factor of 8 -> 112 x 200
        self.h = h
        self.w = w
        self.load_depth_inputs = load_depth_inputs
        self.saved_numpy = saved_numpy

        self.data_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN,
                                                                        std=STD)])

        self.total_n_sequences = 0
        self.scene_paths = sorted(glob(os.path.join(root, mode, '*')))
        self.n_scenes = len(self.scene_paths)

        self.dict_scene_to_n_sequences = {}
        self.dict_scene_to_filenames = {}

        for i, path in enumerate(self.scene_paths):
            filenames = sorted(glob(os.path.join(path, '*_image.jpg')))
            self.dict_scene_to_n_sequences[i] = len(filenames) - seq_len + 1
            self.dict_scene_to_filenames[i] = filenames

            self.total_n_sequences += len(filenames) - seq_len + 1

        # Depth
        if self.load_depth_inputs:
            self.frame_ids = [0, -1, 1]
            self.num_scales = num_scales
            self.interp = Image.ANTIALIAS
            self.img_ext = '.jpg'
            self.loader = pil_loader

            self.resize = {}
            for i in range(self.num_scales):
                s = 2 ** i
                self.resize[i] = transforms.Resize((self.h // s, self.w // s),
                                                   interpolation=self.interp)

            self.load_depth = False
            self.is_train = mode == 'train'

            if self.dataset == 'apollo':
                # Defined as in apollo/utilities/intrinsics.txt
                # fx, 0, Cx / 3384 and fy, 0, Cy / 2710
                self.K = np.array([[0.68101, 0, 0.49830, 0],
                                   [0, 0.85088, 0.49999, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32)
            elif self.dataset in ['kitti', 'kitti_ped', 'davis']:
                self.K = np.array([[0.58, 0, 0.5, 0],
                                   [0, 1.92, 0.5, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32)

    def __len__(self):
        return self.total_n_sequences

    def __getitem__(self, idx):
        """
        Returns
        -------
            data: dict with keys
                img: torch.tensor<float32> (T, 3, H, W)
                instance_seg: torch.tensor<uint8> (T, H, W)
                input_depth: dict with keys:
                    ("color", <frame_id>, <scale>): torch.tensor<float32> (T, 3, H, W)
                        raw colour images
                    ("color_aug", <frame_id>, <scale>): torch.tensor<float32> (T, 3, H, W)
                        augmented colour images
                    ("K", scale) or ("inv_K", scale): torch.tensor<float32> (T, 4, 4)
                        camera intrinsics

                depth: torch.tensor<float32> (T, H, W)                 # Only in nuscenes for now
                position: torch.tensor<float32> (T, MAX_INSTANCES, 3)  # Only in nuscenes for now
                velocity: torch.tensor<float32> (T, MAX_INSTANCES, 3)  # Only in nuscenes for now
                intrinsics: torch.tensor<float32> (T, 3, 3)            # Only in nuscenes for now

        """
        scene_number, position_in_seq = self.get_scene_number(idx)

        img_filenames = self.dict_scene_to_filenames[scene_number][position_in_seq:(position_in_seq + self.seq_len)]

        data = {}
        if self.dataset == 'nuscenes':
            keys = ['img', 'instance_seg', 'depth', 'position', 'velocity', 'intrinsics']
        elif self.dataset in ['kitti', 'kitti_ped', 'davis', 'apollo']:
            keys = ['img', 'instance_seg', 'input_depth']

        for key in keys:
            data[key] = []

        for t in range(self.seq_len):
            data_one_frame = self.get_single_data(img_filenames[t])
            for key in keys:
                if key != 'input_depth':
                    data[key].append(data_one_frame[key])

        if self.load_depth_inputs:
            #  Depth input data
            # Exclude first and last frame, as one past frame and one future frame is needed
            for t in range(1, self.seq_len - 1):
                triplet_img_filename = {-1: img_filenames[t-1],
                                        0: img_filenames[t],
                                        1: img_filenames[t+1]
                                        }
                data['input_depth'].append(self.get_depth_input(triplet_img_filename))

            # Add dummy values for first and last time index
            dummy_input = {key: torch.zeros_like(value) for key, value in data['input_depth'][0].items()}
            data['input_depth'].insert(0, dummy_input)
            data['input_depth'].append(dummy_input)

        # Stack tensor in time dimension
        for key in keys:
            if key != 'input_depth':
                data[key] = torch.stack(data[key], dim=0)
            else:
                if self.load_depth_inputs:
                    input_depth_dict = {}
                    for depth_key in data[key][0].keys():
                        input_depth_dict[depth_key] = torch.stack([data[key][t][depth_key] for t in range(self.seq_len)],
                                                                  dim=0)
                    data[key] = input_depth_dict
        return data

    def get_single_data(self, img_filename):
        base_filename = img_filename[:-len('image.jpg')]
        img = pil_loader(img_filename)
        instance_seg = np.load(base_filename + 'instance_seg.npy')

        if self.dataset == 'nuscenes':
            depth = np.load(base_filename + 'disp.npy')
            position = np.load(base_filename + 'position.npy')
            velocity = np.load(base_filename + 'velocity.npy')
            intrinsics = np.load(base_filename + 'intrinsics.npy')

            # TODO: Remove this check in future
            instance_seg[instance_seg >= MAX_INSTANCES] = 0

        if not self.saved_numpy:
            img, instance_seg, depth, intrinsics = resize_nuscenes_data(img, instance_seg, depth, intrinsics,
                                                                        h_target=self.h, w_target=self.w)

        # Convert to pytorch
        img = self.data_transforms(img)
        instance_seg = torch.from_numpy(instance_seg).to(torch.uint8)

        if self.dataset == 'nuscenes':
            depth = torch.from_numpy(depth).float()
            position = torch.from_numpy(position).float()
            velocity = torch.from_numpy(velocity).float()
            intrinsics = torch.from_numpy(intrinsics).float()

        data_one_frame = {'img': img,
                          'instance_seg': instance_seg,
                          }

        if self.dataset == 'nuscenes':
            data_one_frame['depth'] = depth
            data_one_frame['position'] = position
            data_one_frame['velocity'] = velocity
            data_one_frame['intrinsics'] = intrinsics

        return data_one_frame

    def get_depth_input(self, triplet_img_filename):
        """
        Parameters
        ----------
            triplet_img_filenames: dict
                contains past frame (key -1), current frame (key 0), future frame (key 1) filenames

        Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = False
        do_flip = False

        for i in self.frame_ids:
            inputs[("color", i, -1)] = self.loader(triplet_img_filename[i])

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.w // (2 ** scale)
            K[1, :] *= self.h // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess_depth_input(inputs, color_aug)

        for i in self.frame_ids:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            pass

        return inputs

    def preprocess_depth_input(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.data_transforms(f)
                inputs[(n + "_aug", im, i)] = self.data_transforms(color_aug(f))

    def get_scene_number(self, idx):
        start = 0
        for scene_number in range(self.n_scenes):
            end = start + self.dict_scene_to_n_sequences[scene_number]
            if start <= idx < end:
                position_in_seq = idx - start
                return scene_number, position_in_seq
            start = end

        raise ValueError('Index {} not found in dataset with {} sequences'.format(idx, self.total_n_sequences))


def resize_nuscenes_data(img, instance_seg, depth, intrinsics, h_target, w_target):
    # Resize
    original_h, original_w = instance_seg.shape[-2:]
    resize_scale = h_target / original_h
    h, w = h_target, int(np.ceil(original_w * resize_scale))
    assert (h == h_target) and (w == w_target), 'Mismatch in w: size {} but expected {}'.format(w, w_target)

    img = transforms.Resize((h, w), interpolation=Image.BILINEAR)(img)
    instance_seg = (255 * resize(instance_seg, (N_CLASSES, h, w), order=0, anti_aliasing=None)).astype(np.uint8)
    depth = resize(depth, (h, w), order=1, anti_aliasing=None)
    # Intrinsics
    # If resize scale is different for x and y, need to adapt.
    intrinsics[0, 0] *= resize_scale
    intrinsics[0, 2] *= resize_scale
    intrinsics[1, 1] *= resize_scale
    intrinsics[1, 2] *= resize_scale

    return img, instance_seg, depth, intrinsics


def resize_one_item_multiprocessing(img_fname, new_root, mode, h_target, w_target):
    scene = os.path.basename(os.path.dirname(img_fname))
    save_path = os.path.join(new_root, mode, scene)
    prefix = os.path.basename(img_fname)[:-len('image.jpg')]

    base_filename = img_fname[:-len('image.jpg')]
    img = pil_loader(img_fname)
    instance_seg = np.load(base_filename + 'instance_seg.npy')
    depth = np.load(base_filename + 'disp.npy')
    position = np.load(base_filename + 'position.npy')
    velocity = np.load(base_filename + 'velocity.npy')
    intrinsics = np.load(base_filename + 'intrinsics.npy')

    img, instance_seg, depth, intrinsics = resize_nuscenes_data(img, instance_seg, depth, intrinsics,
                                                                h_target=h_target, w_target=w_target)
    os.makedirs(save_path, exist_ok=True)
    img.save(os.path.join(save_path, prefix + 'image.jpg'))
    np.save(os.path.join(save_path, prefix + 'instance_seg.npy'), instance_seg)
    np.save(os.path.join(save_path, prefix + 'disp.npy'), depth)
    np.save(os.path.join(save_path, prefix + 'position.npy'), position)
    np.save(os.path.join(save_path, prefix + 'velocity.npy'), velocity)
    np.save(os.path.join(save_path, prefix + 'intrinsics.npy'), intrinsics)


def save_dataset_into_disk(h_target=112, w_target=200, root='', mode='train',
                           new_name='debug_112x200'):
    if root[-1] == '/':
        dirname = os.path.dirname(os.path.dirname(root))
    else:
        dirname = os.path.dirname(root)
    new_root = os.path.join(dirname, new_name)
    img_filenames = sorted(glob(os.path.join(root, mode, '*', '*_image.jpg')))

    pool = Pool(cpu_count() - 1)
    for _ in tqdm(pool.imap_unordered(partial(resize_one_item_multiprocessing, new_root=new_root, mode=mode,
                                              h_target=h_target, w_target=w_target),
                                      img_filenames), total=len(img_filenames)):
        pass
