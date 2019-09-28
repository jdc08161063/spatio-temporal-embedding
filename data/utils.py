import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from skimage.transform import resize

from common.constants import MEAN, STD


def crop_and_resize(x, out_size=(128, 256), order=0, crop=True):
    """ Opens and resizes an image as out_size (h, w). Also add option to crop top part of the image so that the ratio
        w/h = 2.0
        Usually resize to (128, 256)
    """
    if isinstance(x, Image.Image):  # PIL Image
        # Crop the top part of the image
        w, h = x.size
        if crop:
            assert (w, h) == (3384, 2710), 'Image size is wrong.'
            x = x.crop((0, h - 1692, w, h))

        if order == 0:
            interpolation = Image.NEAREST
        else:
            interpolation = Image.BILINEAR
        out = transforms.Resize(out_size, interpolation=interpolation)(x)
    elif isinstance(x, np.ndarray):
        h, w = x.shape
        if crop:
            x = x[(h - 1692):, :]
        out = resize(x, out_size, order=order, anti_aliasing=None)
    else:
        raise ValueError('Can resize PIL Image or np.ndarray objects, but received {}'.format(type(x)))

    return out


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def torch_img_to_numpy(img):
    """
    Parameters
    ----------
        img: torch.tensor<float32> (batch_size, seq_len, 3, H, W)

    Returns
    -------
        img_np = np.array<uint8> (batch_size, seq_len, H, W, 3)
    """
    mean_np = np.array(MEAN).reshape((1, 1, 3, 1, 1))
    std_np = np.array(STD).reshape((1, 1, 3, 1, 1))

    img_np = img.detach().cpu().numpy()

    img_np = std_np * img_np + mean_np
    img_np = (255 * img_np).astype(np.uint8)
    img_np = img_np.transpose((0, 1, 3, 4, 2))
    return img_np


def denormalise(x, dimension=5):
    mean = torch.tensor(MEAN)
    std = torch.tensor(STD)

    if dimension == 5:
        mean = mean.view(1, 1, 3, 1, 1)
        std = std.view(1, 1, 3, 1, 1)
    elif dimension == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        raise ValueError('Wrong dimension {}'.format(dimension))

    return std * x + mean