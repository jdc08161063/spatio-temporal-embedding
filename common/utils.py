import os
import sys
import collections
import yaml
import skvideo.io

import numpy as np
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from sklearn.metrics import confusion_matrix

LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'])


def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)


def compute_miou(gt, pred, n_classes=19):
    """ Calculate the mean IOU defined as TP / (TP + FN + FP).
    Parameters
    ----------
        gt: np.array (batch_size, H, W)
        pred: np.array (batch_size, H, W)
    """
    # Compute confusion matrix. IGNORED_ID being equal to 255, it will be ignored.
    cm = confusion_matrix(gt.ravel(), pred.ravel(), np.arange(n_classes))

    # Calculate mean IOU
    miou_dict = {}
    miou = 0
    actual_n_classes = 0
    for l in range(n_classes):
        tp = cm[l, l]
        fn = cm[l, :].sum() - tp
        fp = cm[:, l].sum() - tp
        denom = tp + fn + fp
        if denom == 0:
            iou = float('nan')
        else:
            iou = tp / denom
        if not (np.isnan(iou)):
            miou_dict[LABEL_NAMES[l]] = iou
            miou += iou
            actual_n_classes += 1

    miou /= actual_n_classes

    miou_dict['miou'] = miou
    return miou_dict


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class Logger():
    """ Writes on both terminal and output file."""
    # TODO: add tensoflow logging
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Needed for compatibility
        pass

    def close(self):
        self.log.flush()
        os.fsync(self.log.fileno())
        self.log.close()


def _update_dict_recursive(dict_1, dict_2):
    if isinstance(dict_1, dict) and isinstance(dict_2, collections.Mapping):
        for key, value in dict_2.items():
            dict_1[key] = _update_dict_recursive(dict_1.get(key, None), value)
        return dict_1
    return dict_2


def _ordered_dict_constructor(loader, node):
    pairs = loader.construct_pairs(node)
    res = collections.OrderedDict()
    for key, value in pairs:
        res[key] = _update_dict_recursive(res.get(key, None), value)
    return res


def _ordered_dict_representer(dumper, data):
    return yaml.nodes.MappingNode(yaml.SafeDumper.DEFAULT_MAPPING_TAG,
                                  [(dumper.represent_data(k), dumper.represent_data(v))
                                   for k, v in data.items()])


class _Loader(yaml.SafeLoader):
    def __init__(self, config_path):
        self.config_path = config_path
        super().__init__(open(config_path, 'r'))
        self.yaml_constructors[self.DEFAULT_MAPPING_TAG] = _ordered_dict_constructor
        yaml.SafeDumper.yaml_representers[collections.OrderedDict] = _ordered_dict_representer

    def construct_pairs(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None,
                                   "expected a mapping node, but found %s" % node.id,
                                   node.start_mark)
        pairs = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            if key == "import":
                import_path = os.path.join(os.path.dirname(self.config_path), value)
                if not os.path.isfile(import_path):
                    raise FileNotFoundError('cannot load referenced yml at: {}'.format(import_path))
                imported_config = load_config(import_path)
                for imported_key, imported_value in imported_config.items():
                    pairs.append((imported_key, imported_value))
            else:
                pairs.append((key, value))
        return pairs

    def dispose(self):
        self.stream.close()
        super().dispose()


def _load_config(config_path):
    loader = _Loader(config_path)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def load_config(config_path):
    config = _load_config(config_path)
    config['config_path'] = config_path
    return config


def write_mp4_file(video, output_filename, fps='5'):
    """ Lossless mp4 video creation.
    Parameters
    ----------
    video: list<np.ndarray>
        each array must be (h, w, 3) RGB
    output_filename: str
    fps: str
    """
    video_writer = skvideo.io.FFmpegWriter(output_filename, inputdict={'-r': fps},
                                           outputdict={
        '-vcodec': 'libx264',  #use the h.264 codec
        '-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'veryslow',  #the slower the better compression, in princple, try
                               #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        '-r': fps,
    })
    for j in range(len(video)):
        video_writer.writeFrame(video[j])

    video_writer.close()


def normalise_numpy_image(x):
    x_min = x.min()
    x_max = x.max()

    d = (x_max - x_min) if x_max != x_min else 1e-5
    return (x - x_min) / d
