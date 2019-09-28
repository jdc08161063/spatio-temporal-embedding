import torch
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from matplotlib import gridspec
from visualisation.utils import create_colormap

from common.utils import LABEL_NAMES


def convert_to_pil(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (x * std) + mean
    return transforms.ToPILImage()(x)


####### 
# Cityscapes semantic segmentation visualisation
#######
# Segmentation visualisation
def visualise_sem_seg(img, gt, y, save_filename=''):
    y = y.cpu().data.numpy()
    y = np.transpose(y, (1, 2, 0))
    predicted_labels = np.argmax(y, axis=-1)
    gt = gt.cpu().data.numpy()

    colormap = create_colormap()
    plt.figure(figsize=(20, 10))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(convert_to_pil(img.cpu()))
    plt.axis('off')
    plt.title('Image')

    plt.subplot(grid_spec[1])
    plt.imshow(colormap[gt])
    plt.axis('off')
    plt.title('Ground truth seg')

    plt.subplot(grid_spec[2])
    plt.imshow(colormap[predicted_labels])
    plt.axis('off')
    plt.title('Predicted seg')

    unique_labels = np.unique(predicted_labels)
    ax = plt.subplot(grid_spec[3])
    # Legend
    full_color_map = colormap[np.arange(len(LABEL_NAMES))[:, None]]

    plt.imshow(
        full_color_map[unique_labels].astype(np.uint8))
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid(False)

    if save_filename:
        plt.savefig(save_filename)
        plt.close()
    else:
        plt.show()


def compare_bbox_instance_seg(img, nuscenes_box, instance_seg):
    idx = 0
    _, ax = plt.subplots(1, 1, figsize=(9, 16))
    # Show image.
    ax.imshow(img)

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=2)
            prev = corner

    corresponding_box = nuscenes_box
    x_min = corresponding_box['x1']
    x_max = corresponding_box['x2']
    y_min = corresponding_box['y1']
    y_max = corresponding_box['y2']
    {'x1': x_min, 'x2': x_max, 'y1': y_min, 'y2': y_max}
    bounding_box_2d = np.array([[x_min, y_min],
                                [x_min, y_max],
                                [x_max, y_max],
                                [x_max, y_min]])

    draw_rect(bounding_box_2d, 'b')
    # draw_rect(corners.T[:4], color)
    plt.show()
    plt.figure(figsize=(9, 16))
    plt.imshow((instance_seg[idx]).squeeze(), cmap='gray')
    plt.show()


def visualise_nuscenes_3D(nusc):
    SENSOR = 'CAM_FRONT'

    scene = nusc.scene[0]
    sample_token = scene['first_sample_token']
    count = 0

    while sample_token:
        print(sample_token)
        sample = nusc.get('sample', sample_token)
        data_token = sample['data'][SENSOR]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(data_token)

        nusc.render_sample_data(data_token)
        plt.show()
        sample_token = sample['next']
        count += 1
