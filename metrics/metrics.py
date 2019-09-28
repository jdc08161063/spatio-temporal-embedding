import sys
import json

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.detection import NuScenesEval
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.config import config_factory

from common.utils import cummean


class MotionMetrics():
    def __init__(self, config, tensorboard, min_recall=0.1, min_precision=0.1):
        self.config = config
        self.tensorboard = tensorboard
        self.threshold = self.config['metrics_threshold']
        self.min_recall = min_recall
        self.min_precision = min_precision

        self.metrics = {}

        self.tp = None  # True positives
        self.fp = None  # False positives
        self.conf = None  # Confidence score
        self.n_positive = 0  # Number of ground truth instances seen so far

        # Position/velocity error for true positives only
        self.match_data = None

        # MOTS metrics
        self.tp_mots = None
        self.fp_mots = None
        self.fn_mots = None
        self.soft_tp_mots = None
        self.n_switches = None

        self.reset()

    def update(self, batch, output):
        """
        Parameters
        ----------
            batch: dict with keys:
                instance_seg: torch.tensor (b, seq_len, N_CLASSES, h, w)
                position: torch.tensor (b, seq_len, MAX_INSTANCES, 3)
                velocity: torch.tensor (b, seq_len, MAX_INSTANCES, 3)

            output: dict with keys:
                instance_seg: torch.tensor (b, seq_len, N_CLASSES, h, w)
                position: torch.tensor (b, seq_len, MAX_INSTANCES, 3)
                velocity: torch.tensor (b, seq_len, MAX_INSTANCES, 3)
        """
        receptive_field = self.config['receptive_field']
        if not self.config['instance_loss']:
            return
        batch_keys = ['img', 'instance_seg']
        output_keys = ['instance_seg']

        if self.config['motion_loss']:
            batch_keys += ['position', 'velocity']
            output_keys += ['position', 'velocity']

        batch_np = {key: batch[key].detach().cpu().numpy() for key in batch_keys}
        output_np = {key: output[key].detach().cpu().numpy() for key in output_keys}

        b, seq_len = batch_np['img'].shape[:2]

        dict_prev_id = {}
        for i in range(b):
            for t in range(receptive_field - 1, seq_len):
                pred_unique_ids = np.unique(output_np['instance_seg'][i, t])[1:]
                gt_unique_ids = np.unique(batch_np['instance_seg'][i, t])[1:]
                self.n_positive += len(gt_unique_ids)

                taken_gt_ids = set()
                used_pred_ids = set()
                for id in pred_unique_ids:
                    mask = output_np['instance_seg'][i, t] == id
                    best_iou = 0
                    best_gt_id = None
                    for gt_id in gt_unique_ids:
                        if gt_id in taken_gt_ids:
                            continue

                        gt_mask = batch_np['instance_seg'][i, t] == gt_id
                        inter = (mask & gt_mask).sum()
                        union = (mask | gt_mask).sum()
                        iou = inter / union
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_id = gt_id

                    conf = np.random.random()
                    if best_iou > self.threshold:
                        self.tp.append(1)
                        self.fp.append(0)
                        taken_gt_ids.add(best_gt_id)
                        used_pred_ids.add(id)

                        if self.config['motion_loss']:
                            # 2D velocity error
                            self.match_data['vel_err'].append(calc_vel_err(output_np['velocity'][i, t, gt_unique_ids],
                                                                           batch_np['velocity'][i, t, gt_unique_ids]))
                            self.match_data['pos_err'].append(np.linalg.norm((output_np['position'][i, t, gt_unique_ids]
                                                                              - batch_np['position'][i, t, gt_unique_ids])))
                            self.match_data['conf'].append(conf)

                        self.tp_mots += 1
                        self.soft_tp_mots += best_iou

                        if best_gt_id in dict_prev_id and id != dict_prev_id[best_gt_id]:
                            self.n_switches += 1
                        dict_prev_id[best_gt_id] = id

                    else:
                        self.tp.append(0)
                        self.fp.append(1)
                    self.conf.append(conf)  # TODO: add confidence score

                self.fp_mots += len(set(pred_unique_ids).difference(used_pred_ids))
                self.fn_mots += len(set(gt_unique_ids).difference(taken_gt_ids))

    def evaluate(self, global_step, mode):
        if not self.config['instance_loss']:
            return 0.0
        if len(self.tp) == 0:
            print('No accumulated metrics')
            self.reset()
            return 0

        # Sort by decreasing confidence score
        self.conf = np.array(self.conf)
        indices = np.argsort(-self.conf)
        self.tp = np.array(self.tp)[indices]
        self.fp = np.array(self.fp)[indices]
        self.conf = self.conf[indices]

        if self.config['motion_loss']:
            match_data_indices = np.argsort(-np.array(self.match_data['conf']))
            for key in self.match_data.keys():
                if key == 'conf':
                    continue
                self.match_data[key] = np.array(self.match_data[key])[match_data_indices]
            self.match_data['conf'] = np.array(self.match_data['conf'])[match_data_indices]

        # Compute Average Precision
        self.tp = np.cumsum(self.tp)
        self.fp = np.cumsum(self.fp)
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / max(1, self.n_positive)

        # Interpolate to a equally spaced recall values [0-1] with 0.01 increment
        recall_interp = np.linspace(0, 1, 101)
        precision = np.interp(recall_interp, recall, precision, right=0)
        self.conf = np.interp(recall_interp, recall, self.conf, right=0)

        if self.config['motion_loss']:
            for key in self.match_data.keys():
                if key == 'conf':
                    continue
                tmp = cummean(self.match_data[key])
                self.match_data[key] = np.interp(self.conf[::-1], self.match_data['conf'][::-1], tmp[::-1])[::-1]

        # Average Precision: area under the precision/recall curve for recall and precision over 10%
        self.metrics['ap'] = calc_ap(precision, self.min_recall, self.min_precision)
        if self.config['motion_loss']:
            self.metrics['vel_err'] = calc_tp(self.match_data['vel_err'], self.conf, self.min_recall)
            self.metrics['pos_err'] = calc_tp(self.match_data['pos_err'], self.conf, self.min_recall)

        # MOTS metrics
        self.metrics['motsa'] = (self.tp_mots - self.fp_mots - self.n_switches) / max(1, self.n_positive)
        self.metrics['motsp'] = self.soft_tp_mots / max(1, self.tp_mots)
        self.metrics['smotsa'] = (self.soft_tp_mots - self.fp_mots - self.n_switches) / max(1, self.n_positive)
        self.metrics['n_switches'] = self.n_switches
        self.metrics['tp_mots'] = self.tp_mots
        self.metrics['fp_mots'] = self.fp_mots
        self.metrics['fn_mots'] = self.fn_mots
        self.metrics['soft_tp_mots'] = self.soft_tp_mots
        self.metrics['n_positive'] = self.n_positive
        self.metrics['recall'] = self.tp_mots / max(1, self.n_positive)
        self.metrics['precision'] = self.tp_mots / max(1, (self.tp_mots + self.fp_mots))

        for key, value in self.metrics.items():
            print('{}: {:.3f}'.format(key, value))
            self.tensorboard.add_scalar(mode + '/' + key, value, global_step)

        metric_score = self.score()
        self.reset()
        return metric_score

    def score(self):
        return self.metrics['motsa']

    def reset(self):
        self.metrics = {}

        self.tp = []
        self.fp = []
        self.conf = []
        self.n_positive = 0
        self.match_data = {'vel_err': [],
                           'pos_err': [],
                           'conf': [],
                           }

        self.tp_mots = 0
        self.fp_mots = 0
        self.fn_mots = 0
        self.soft_tp_mots = 0
        self.n_switches = 0


def calc_vel_err(pred_vel, gt_vel):
    # 2D velocity error
    vel_err = np.linalg.norm(pred_vel[:, [0, 2]] - gt_vel[:, [0, 2]])
    return vel_err


def calc_ap(precision_interp, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(precision_interp)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(tp_metric, confidence, min_recall: float) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = max_recall_ind(confidence)  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(tp_metric[first_ind: last_ind + 1]))  # +1 to include error at max recall


def max_recall_ind(confidence):
    """ Returns index of max recall achieved. """

    # Last instance of confidence > 0 is index of max achieved recall.
    non_zero = np.nonzero(confidence)[0]
    if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
        max_recall_ind = 0
    else:
        max_recall_ind = non_zero[-1]

    return max_recall_ind


if __name__ == '__main__':
    result_path_ = ''
    output_dir_ = ''
    eval_set_ = 'val'
    dataroot_ = ''
    version_ = 'v1.0-trainval'
    config_path = ''
    plot_examples_ = 0
    render_curves_ = False
    verbose_ = True

    if config_path == '':
        cfg_ = config_factory('cvpr_2019')
    else:
        with open(config_path, 'r') as f:
            cfg_ = DetectionConfig.deserialize(json.load(f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = NuScenesEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                             output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)