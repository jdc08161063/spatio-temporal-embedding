import os

import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from glob import glob

from torch.utils.data import Dataset, DataLoader

from common.constants import MEAN, STD

DATA_ROOT = ''


class KittiMaskDataset(Dataset):
    def __init__(self, split='train', data_transforms=None):
        super().__init__()
        self.data_transforms = data_transforms

        self.img_filenames = sorted(glob(os.path.join(DATA_ROOT, split, '*', '*_image.jpg')))
        self.label_filenames = sorted(glob(os.path.join(DATA_ROOT, split, '*', '*_instance_seg.npy')))

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        """
        Returns
        -------
            img: torch Tensor of shape (H, W, 3)
            label: np.array of shape (H, W)
        """
        img = Image.open(self.img_filenames[idx])
        label = np.load(self.label_filenames[idx])

        if self.data_transforms:
            img = self.data_transforms(img)
        else:
            img = transforms.ToTensor()(img)

        # Shape (1, h, w)
        # Binary mask
        label = (torch.from_numpy(label[0]) > 0).long()
        return img, label


def get_kitti_mask_dataloaders(batch_size=2):
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=MEAN,
                                                               std=STD)])

    train_dataset = KittiMaskDataset(split='train', data_transforms=data_transforms)
    val_dataset = KittiMaskDataset(split='val', data_transforms=data_transforms)
    train_iterator = DataLoader(train_dataset, batch_size, shuffle=True)
    val_iterator = DataLoader(val_dataset, batch_size, shuffle=False)
    print('Train size: {}'.format(len(train_dataset)))
    print('Val size: {}'.format(len(val_dataset)))

    return train_iterator, val_iterator
