import sys
sys.path.append('')

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.pylab import cm
from PIL import Image
from sklearn.decomposition import PCA

from nuscenes.utils.geometry_utils import view_points

DEFAULT_COLORMAP = cm.magma


def create_colormap():
    """Creates a colormap for visualisation of instances.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    def bit_get(val, idx):
        """Gets the bit value.
        Args:
        val: Input value, int or numpy int array.
        idx: Which bit of the input val.
        Returns:
        The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def hex_to_rgb(rgb_hex_str):
    """ converts string '0xFFFFFF' to a list of RGB values. """
    rgb_int = int(rgb_hex_str, 16)
    r = rgb_int // (256 * 256)
    g = rgb_int // 256 % 256
    b = rgb_int % 256 % 256
    return [r, g, b]


def apply_colormap(image, cmap=DEFAULT_COLORMAP, autoscale=False):
    """
    Applies a colormap to the given 1 or 2 channel numpy image. if 2 channel, must be 2xHxW. Returns a HxWx3 numpy image
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if image.ndim == 3:
            image = image[0]
        # grayscale scalar image
        if autoscale:
            image = _normalise(image)
        return cmap(image)[:, :, :3]
    if image.shape[0] == 3:
        # normalise rgb channels
        if autoscale:
            image = _normalise(image)
        return np.transpose(image, axes=[1, 2, 0])
    raise Exception('Image must be 1, 2 or 3 channel to convert to colormap (CxHxW)')


def _normalise(image):
    lower = np.min(image)
    delta = np.max(image) - lower
    if delta == 0:
        delta = 1
    image = (image.astype(np.float32) - lower) / delta
    return image


def heatmap_image(image, cmap=DEFAULT_COLORMAP, autoscale=True, output_pil=False):
    """
    Colourise a 1 or 2 channel image with a colourmap.
    """
    image_cmap = apply_colormap(image, cmap=cmap, autoscale=autoscale)
    if output_pil:
        image_cmap = np.uint8(image_cmap * 255)
        return Image.fromarray(image_cmap)
    return image_cmap


def image_to_tensor(pic):
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        if isinstance(img, torch.DoubleTensor):
            return img.float()
        return img

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    return img


def convert_figure_numpy(figure):
    """ Convert figure to numpy image """
    figure_np = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    figure_np = figure_np.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return figure_np


def plot_labels_on_image(img, instance_seg, position=None, velocity=None, intrinsics=None, dpi=100,
                         alpha=0.8, id_legend=True):
    """
    Parameters
    ----------
        img_copy: np.array shape (H, W, 3)
        instance_seg: np.array shape (N_CLASSES, H, W)
        position: np.array shape (MAX_INSTANCES, 3)
        velocity: np.array shape (MAX_INSTANCES, 3)
        intrinsics: np.array shape (3, 3)
    """
    img_copy = img.copy()
    height, width = img_copy.shape[0:2]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.gca()

    # Overlay instance segmentation on rgb image
    colormap = create_colormap()
    instance_seg = instance_seg.squeeze(0)  # TODO: N_CLASSES
    mask = instance_seg != 0
    img_copy[mask] = ((1-alpha) * img_copy[mask] + alpha * colormap[instance_seg][mask]).astype(np.uint8)

    # Print all ids
    unique_ids = np.unique(instance_seg)
    if id_legend:
        ax.plot([], [], ' ', label='IDs: ' + ', '.join([str(x) for x in unique_ids[1:]]))
        ax.legend(loc='upper left', prop={'size': 8})
    # Plot image
    ax.set_axis_off()
    ax.imshow(img_copy)
    if position is not None:
        # Convert position from camera reference frame to image plane
        image_position = view_points(position.T, intrinsics, True)[:2]
        unique_ids = np.unique(instance_seg)
        for inst_id in unique_ids[1:]:
            inst_id = inst_id - 1
            col, row = image_position[:, inst_id]

            text = 'xyz: {:.1f}/{:.1f}/{:.1f}\nv_xyz: {:.1f}/{:.1f}/{:.1f}'.format(*position[inst_id],
                                                                                   *velocity[inst_id])
            text_plot = ax.text(col, row, text, fontsize=5, fontweight='bold', color='black')
            text_plot.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=colormap[inst_id + 1] / 255))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.draw()
    fig_np = convert_figure_numpy(fig)
    plt.close('all')

    return fig_np


def plot_embedding_clustering(y, instance_seg, mask, config, dpi=100):
    """
    Parameters
    ----------
        y: np.array shape (emb_dim, H, W)
        instance_seg: np.array shape (N_CLASSES, H, W)
        mask: np.array shape(H, W)
    """
    height, width = y.shape[1:]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.gca()

    y = np.transpose(y, (1, 2, 0))
    instance_seg = instance_seg.squeeze(0)  # TODO: N_CLASSES
    mask = mask.astype(np.bool)

    colormap = create_colormap()

    try:
        if mask.sum() > 0:
            y = y[mask]
            instance_seg = instance_seg[mask]

            pca = PCA(n_components=2)
            pca.fit(y)
            #print('Explained variance: {}'.format(pca.explained_variance_ratio_))

            y_two_d = pca.transform(y)
            for id in np.unique(instance_seg):
                if id == 0:
                    continue
                mask_id = (instance_seg == id)
                y1 = y_two_d[:, 0][mask_id]
                y2 = y_two_d[:, 1][mask_id]

                ax.scatter(y1, y2, c=(colormap[id] / 255).reshape((1, 3)), alpha=0.2)
                intra_cluster = plt.Circle((y1.mean(), y2.mean()), config['delta_v'], color='black', linestyle='--', linewidth=2,
                                           fill=False)
                inter_cluster = plt.Circle((y1.mean(), y2.mean()), config['delta_d'], color='black', linestyle='--', linewidth=2,
                                           alpha=0.5, fill=False)
                ax.add_artist(intra_cluster)
                ax.add_artist(inter_cluster)
    except:
        pass

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.draw()
    fig_np = convert_figure_numpy(fig)
    plt.close('all')

    return fig_np


def compute_pixel_barycenter(instance_seg, id):
    """ Compute the pixel barycenter of and instance.

    Parameters
    ----------
        instance_seg: np.ndarray<int> (height, width)
        id: int
            considered id

    Returns
    -------
        barycenter: np.ndarray<float> (2)
            barycenter of the instance in pixel space ie axis are height and width.
    """
    height, width = instance_seg.shape
    mgrid = np.moveaxis(np.mgrid[:height, :width], source=0, destination=-1)
    instance_mask = (instance_seg == id)
    pixel_coords = mgrid[instance_mask]
    return pixel_coords.mean(axis=0)
