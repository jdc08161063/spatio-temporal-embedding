import torch

import numpy as np
import torch.nn as nn

from collections import deque
from sklearn.cluster import MeanShift
from scipy.ndimage.morphology import binary_dilation, binary_erosion

from data.consistency import make_ids_consecutives, enforce_consistency_centers, \
    increment_life_clusters
from common.constants import MAX_INSTANCES_SCENE, N_CLASSES, BANDWIDTH, CLUSTERING_COST_THRESHOLD, CLUSTER_MEAN_LIFE


def print_model_spec(model, name=''):
    n_parameters = count_n_parameters(model)
    n_trainable_parameters = count_n_parameters(model, only_trainable_parameters=True)
    print('Model {}: {} parameters of which {} are trainable'.format(name, n_parameters, n_trainable_parameters))


def count_n_parameters(model, only_trainable_parameters=False):
    if only_trainable_parameters:
        n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        n_parameters = sum([p.numel() for p in model.parameters()])
    return n_parameters


def require_grad(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode


class eval_mode:
    def __init__(self, module):
        if not isinstance(module, nn.Module):
            raise TypeError("test_mode can only handle neural.Module, got {}".format(type(module)))
        self.module = module
        self.train_mode = None

    def __enter__(self):
        self.train_mode = self.module.training
        self.module.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module.train(self.train_mode)


def cluster(y, mask, receptive_field):
    """ Cluster the instances in the embedding space.

    Parameters
    ----------
        y: np.array (batch_size, T, emb_dim, H, W)
        mask: np.array (batch_size, T, H, W)

    Returns
    -------
        n_clusters: np.array (batch_size,)
        pred_instance_seg: np.array (batch_size, T, N_CLASSES, H, W)
        cluster_centers: np.array (batch_size, MAX_INSTANCES, emb_dim)
    """
    batch_size, seq_len, emb_dim, h, w = y.shape
    n_clusters = np.zeros((batch_size,), dtype=np.uint8)
    pred_instance_seg = np.zeros((batch_size, seq_len, N_CLASSES, h, w), dtype=np.uint8)
    cluster_centers = np.zeros((batch_size, MAX_INSTANCES_SCENE, emb_dim), dtype=np.float32)

    y = np.transpose(y, (0, 1, 3, 4, 2))
    mask = mask.astype(np.bool)

    for i in range(batch_size):
        try:
            ms = MeanShift(BANDWIDTH, bin_seeding=True)
            # Cluster everything, but the background
            if mask[i, (receptive_field - 1):].sum() > 0:
                ms.fit(y[i, (receptive_field - 1):][mask[i, (receptive_field - 1):]])

                n_clusters[i] = len(ms.cluster_centers_)
                pred_instance_seg[i, (receptive_field - 1):, 0][mask[i, (receptive_field - 1):]] = ms.labels_ + 1

                cluster_centers[i, :n_clusters[i]] = ms.cluster_centers_
        except:
            pass

    return n_clusters, pred_instance_seg, cluster_centers


def cluster_causal(y, mask, receptive_field, accumulate_embeddings=False):
    """ Cluster the instances in the embedding space, in a causal way.
        Assign ids by comparing embedding centers.

    Parameters
    ----------
        y: np.array (batch_size, T, emb_dim, H, W)
        mask: np.array (batch_size, T, H, W)
        COST_THRESHOLD: float
            euclidean distance threshold between the embedding centers of two frames

    Returns
    -------
        n_clusters: np.array (batch_size, T)
        pred_instance_seg: np.array (batch_size, T, N_CLASSES, H, W)
        cluster_centers: np.array (batch_size, T, MAX_INSTANCES, emb_dim)
    """
    batch_size, seq_len, emb_dim, h, w = y.shape
    n_clusters = np.zeros((batch_size, seq_len), dtype=np.uint8)
    pred_instance_seg = np.zeros((batch_size, seq_len, N_CLASSES, h, w), dtype=np.uint8)
    cluster_centers = np.zeros((batch_size, seq_len, MAX_INSTANCES_SCENE, emb_dim), dtype=np.float32)

    if accumulate_embeddings:
        accumulated_y = deque(maxlen=seq_len)

    y = np.transpose(y, (0, 1, 3, 4, 2))
    mask = mask.astype(np.bool)

    for i in range(batch_size):
        available_keys = set(range(1, MAX_INSTANCES_SCENE))  # max of 256 instances
        dict_centers = {}
        for t in range(receptive_field - 1, seq_len):
            try:
                mask_t = mask[i, t]
                ms = MeanShift(BANDWIDTH, bin_seeding=True)
                if mask_t.sum() > 0:
                    if accumulate_embeddings:
                        accumulated_y.append(y[i, t][mask_t])
                        ms_inputs = np.concatenate(accumulated_y, axis=0)
                    else:
                        ms_inputs = y[i, t][mask_t]
                    ms.fit(ms_inputs)
                    ms_start_index = len(ms_inputs) - len(y[i, t][mask_t])
                    instance_seg_t = (ms.labels_ + 1)[ms_start_index:]

                    full_instance_seg_t = np.zeros_like(mask_t).astype(np.uint8)
                    full_instance_seg_t[mask_t] = instance_seg_t

                    # valid_ids = [id for id in np.unique(full_instance_seg_t) if
                    #              (full_instance_seg_t == id).sum() > MIN_PIXEL_THRESHOLD]
                    # full_instance_seg_t[~np.isin(full_instance_seg_t, valid_ids)] = 0

                    mask_t = full_instance_seg_t > 0
                    if mask_t.sum() > 0:
                        instance_seg_t = full_instance_seg_t[mask_t]
                        # Ids must be consecutive
                        instance_seg_t = make_ids_consecutives(instance_seg_t)

                        if accumulate_embeddings:
                            centers = []
                            for id in np.unique(instance_seg_t):
                                full_instance_seg_t = np.zeros_like(mask_t).astype(np.uint8)
                                full_instance_seg_t[mask_t] = instance_seg_t
                                mask_id = (full_instance_seg_t == id)
                                centers.append(y[i, t][mask_id].mean(axis=0))
                            centers = np.stack(centers, axis=0)
                        else:
                            centers = ms.cluster_centers_

                        consistent_instance_seg_t, available_keys, dict_centers = enforce_consistency_centers(
                            instance_seg_t, centers, dict_centers, available_keys,
                            cost_threshold=CLUSTERING_COST_THRESHOLD, cluster_mean_life=CLUSTER_MEAN_LIFE)

                        n_clusters[i, t] = len(centers)
                        pred_instance_seg[i, t, 0][mask_t] = consistent_instance_seg_t
                        cluster_centers[i, t, :n_clusters[i, t]] = centers  # Note that as such the centers are not
                    else:
                        increment_life_clusters(dict_centers, CLUSTER_MEAN_LIFE)
            except:
                pass


    return n_clusters, pred_instance_seg, cluster_centers


def cluster_frame_by_frame(y, mask, receptive_field=1):
    """ Cluster the instances in the embedding space (for the one-frame model).

    Parameters
    ----------
        y: np.array (batch_size, T, emb_dim, H, W)
        mask: np.array(batch_size, T, H, W)

    Returns
    -------
        n_clusters: np.array (batch_size, T)
        pred_instance_seg: np.array (batch_size, T, N_CLASSES, H, W)
        cluster_centers: np.array (batch_size, T, MAX_INSTANCES, emb_dim)
    """
    batch_size, seq_len, emb_dim, h, w = y.shape
    n_clusters = np.zeros((batch_size, seq_len), dtype=np.uint8)
    pred_instance_seg = np.zeros((batch_size, seq_len, N_CLASSES, h, w), dtype=np.uint8)
    cluster_centers = np.zeros((batch_size, seq_len, MAX_INSTANCES_SCENE, emb_dim), dtype=np.float32)

    y = np.transpose(y, (0, 1, 3, 4, 2))
    mask = mask.astype(np.bool)

    for i in range(batch_size):
        for t in range(seq_len):
            try:
                ms = MeanShift(BANDWIDTH, bin_seeding=True)
                # Cluster everything, but the background
                if mask[i, t].sum() > 0:
                    ms.fit(y[i, t][mask[i, t]])

                    n_clusters[i, t] = len(ms.cluster_centers_)
                    pred_instance_seg[i, t, 0][mask[i, t]] = ms.labels_ + 1

                    cluster_centers[i, t, :n_clusters[i, t]] = ms.cluster_centers_
            except:
                pass

    return n_clusters, pred_instance_seg, cluster_centers


def cluster_groundtruth_frame(y, gt_instance_seg, bandwidth=1.5):
    """
    Parameters
    ----------
        y: torch.Tensor<float32> (n, emb_dim)
        gt_instance_seg: torch.Tensor<uint8> (n)
        bandwidth: float
            cluster threshold

    Returns
    -------
        prediction: torch.Tensor<uint8> (n)
        mean: torch.Tensor<float32> (n_instances, emb_dim)
    """
    emb_dim = y.shape[0]
    prediction = torch.zeros_like(gt_instance_seg).byte()
    mean = []

    if (gt_instance_seg > 0).sum() == 0:
        return prediction, torch.zeros((0, emb_dim))

    instance_id = 0
    for current_id in torch.unique(gt_instance_seg):
        mask_id = gt_instance_seg == current_id
        if current_id == 0:
            continue

        current_mean = y[mask_id].mean(dim=0)
        threshold_mask = (torch.norm(y - current_mean, p=2, dim=1) < bandwidth)
        if threshold_mask.sum() > 0:
            instance_id += 1
            prediction[threshold_mask] = instance_id
            mean.append(current_mean)

    prediction[gt_instance_seg == 0] = 0
    if len(mean) > 0:
        mean = torch.stack(mean, dim=0)
    else:
        torch.zeros((0, emb_dim))
    return prediction, mean


def cluster_groundtruth(y, gt_instance_seg, mask, receptive_field=1, bandwidth=1.5):
    """ Cluster the instances in the embedding space, in a causal way.
        Assign ids by comparing embedding centers.

    Parameters
    ----------
        y: torch.Tensor<float32>  (batch_size, T, emb_dim, H, W)
        gt_instance_seg: torch.Tensor<uint8> (batch_size, T, H, W)

    Returns
    -------
        n_clusters: np.array (batch_size, T)
        pred_instance_seg: np.array (batch_size, T, N_CLASSES, H, W)
        cluster_centers: np.array (batch_size, T, MAX_INSTANCES, emb_dim)
    """
    batch_size, seq_len, emb_dim, h, w = y.shape
    n_clusters = np.zeros((batch_size, seq_len), dtype=np.uint8)
    pred_instance_seg = np.zeros((batch_size, seq_len, N_CLASSES, h, w), dtype=np.uint8)
    cluster_centers = np.zeros((batch_size, seq_len, MAX_INSTANCES_SCENE, emb_dim), dtype=np.float32)

    y = y.permute((0, 1, 3, 4, 2))  # (h, w, emb_dim)
    for i in range(batch_size):
        available_keys = set(range(1, MAX_INSTANCES_SCENE))  # max of 256 instances
        dict_centers = {}
        for t in range(receptive_field - 1, seq_len):
            mask_t = mask[i, t]
            y_t = y[i, t][mask_t]
            gt_t = gt_instance_seg[i, t].squeeze(0)[mask_t]
            pred_t, mean_t = cluster_groundtruth_frame(y_t, gt_t, bandwidth=bandwidth)
            pred_t = pred_t.detach().cpu().numpy()
            mean_t = mean_t.detach().cpu().numpy()
            mask_t = mask_t.detach().cpu().numpy().astype(np.bool)

            full_instance_seg_t = np.zeros_like(mask_t).astype(np.uint8)
            full_instance_seg_t[mask_t] = pred_t

            mask_t = full_instance_seg_t > 0
            if mask_t.sum() > 0:
                instance_seg_t = full_instance_seg_t[mask_t]
                # Ids must be consecutive
                instance_seg_t = make_ids_consecutives(instance_seg_t)

                centers = []
                for id in np.unique(instance_seg_t):
                    full_instance_seg_t = np.zeros_like(mask_t).astype(np.uint8)
                    full_instance_seg_t[mask_t] = instance_seg_t
                    mask_id = (full_instance_seg_t == id)
                    centers.append(y[i, t].detach().cpu().numpy()[mask_id].mean(axis=0))
                centers = np.stack(centers, axis=0)

                consistent_instance_seg_t, available_keys, dict_centers = enforce_consistency_centers(
                    instance_seg_t, centers, dict_centers, available_keys,
                    cost_threshold=CLUSTERING_COST_THRESHOLD, cluster_mean_life=CLUSTER_MEAN_LIFE, verbose=True)

                pred_instance_seg[i, t, 0][mask_t] = consistent_instance_seg_t
                n_clusters[i, t] = len(mean_t)
                cluster_centers[i, t, :len(mean_t)] = mean_t
            else:
                increment_life_clusters(dict_centers, CLUSTER_MEAN_LIFE)

    return n_clusters, pred_instance_seg, cluster_centers


def cluster_meanshift_frame(y, mask, bandwidth=1.5):
    """
    Parameters
    ----------
        y: torch.Tensor<float32> (emb_dim, h, w)
        mask: torch.Tensor<uint8> (h, w)
        bandwidth: int
            cluster threshold

    Returns
    -------
        prediction: torch.Tensor<uint8> (h, w)
    """
    device = y.device
    prediction = torch.zeros_like(mask).byte()

    if mask.byte().sum() == 0:
        return prediction

    unclustered = mask.clone().detach().byte()
    instance_id = 0
    y = y.permute((1, 2, 0))  # (h, w, emb_dim)
    counter = 0
    while unclustered.sum() > 100 and counter < 20:
        y_masked = y[unclustered]

        #  Select random pixel
        mean = y_masked[torch.randint(len(y_masked), size=(1,)).item()]
        new_mean = mean_shift(mean, y_masked, bandwidth=bandwidth)

        mean_iterations = 0
        while (torch.norm(new_mean - mean, p=2) > 0.0001) and mean_iterations < 100:
            mean_iterations += 1
            mean = new_mean
            new_mean = mean_shift(mean, y_masked, bandwidth=bandwidth)

        if mean_iterations < 100:
            #  Assign instance id
            instance_id += 1
            # Threshold around mean
            threshold_mask = (torch.norm(y - new_mean, p=2, dim=2) < bandwidth)

            # Do not create a new instance if there is significant overlap with an already existing instance
            intersection = (threshold_mask * (prediction > 0)).sum()
            intersection /= threshold_mask.sum()
            if intersection.item() < 0.5:
                # Remove already assigned pixels
                threshold_mask *= unclustered
                # Erosion followed by dilation to remove noise
                threshold_mask_tmp = binary_erosion(threshold_mask.detach().cpu().numpy())
                threshold_mask_tmp = binary_dilation(threshold_mask_tmp)
                threshold_mask_tmp = torch.from_numpy(threshold_mask_tmp).to(device)
                prediction[threshold_mask_tmp] = instance_id
                counter = 0

            # Update unclustered
            unclustered[threshold_mask] = 0
        else:
            counter += 1

    # Remove noise
    prediction_tmp = torch.zeros_like(prediction)
    instance_id = 0
    for current_id in torch.unique(prediction):
        if current_id == 0:
            continue
        if (prediction == current_id).sum().item() > 10:
            instance_id += 1
            prediction_tmp[prediction == current_id] = instance_id

    prediction = prediction_tmp

    return prediction


def mean_shift(mean, y_masked, bandwidth=1.5):
    """
    Parameters
    ----------
        mean: torch.Tensor<float32> (emb_dim,)
        y_masked: torch.Tensor<float32> (n, emb_dim)

    Returns
    -------
        new_mean: torch.Tensor<float32> (emb_dim,)
    """
    valid = torch.norm(y_masked - mean, p=2, dim=1) < bandwidth
    if valid.sum() > 0:
        return y_masked[valid].mean(dim=0)
    else:
        return mean
