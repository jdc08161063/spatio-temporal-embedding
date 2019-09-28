import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment

from common.constants import MAX_INSTANCES_SCENE


def temporally_align_sequence(instance_seg, iou_threshold=0.1):
    """
    Parameters
    ----------
        instance_seg: np.ndarray<np.uint8> (batch_size, seq_len, N_CLASSES, height, width)
    """
    aligned_instance_seg = np.zeros_like(instance_seg)
    batch_size, seq_len = instance_seg.shape[:2]

    for i in range(batch_size):
        available_keys = set(range(1, MAX_INSTANCES_SCENE))  # max of 256 instances
        dict_ids = None
        prev_instance_seg_t = None
        for t in range(seq_len):
            instance_seg_t = instance_seg[i, t]

            # The algorithm only works when for each frame, instance ids are in [0, max_n_instance[
            instance_seg_t = make_ids_consecutives(instance_seg_t)
            # Enforce consistency
            consistent_instance_seg_t, available_keys, dict_ids = enforce_consistency(instance_seg_t,
                                                                                      prev_instance_seg_t,
                                                                                      available_keys, dict_ids,
                                                                                      cost_threshold=(1 - iou_threshold))
            aligned_instance_seg[i, t] = consistent_instance_seg_t
            prev_instance_seg_t = instance_seg_t

    return aligned_instance_seg


def enforce_consistency(inst_seg, prev_inst_seg, available_keys, dict_ids, cost_threshold=0.99, use_barycenter=False,
                        centers=None, prev_centers=None):
    """
    TODO: remove center barycenter parameters
    Make the instance ids consistent the following way:

    Step 1: For each instance in the current frame, try to assign it to an instance in the previous frame.
            If one instance is assigned, but with cost 1.0, ignore it (as there is no overlap at all)
    Step 2: REMOVED FOR NOW. For the remaining instances, compare to all the instances that previously existed using
            barycenter distance.
            Match them if the distance is below some threshold. The barycenter can only be used to bridge 3 frames.
            ie usually, one frame in the middle where the instance disappears
    Step 3: If there still is some unassigned instances, assign to a new unique id

    Parameters
    ----------
    inst_seg: np.ndarray<uint8> (h, w)
    prev_inst_seg: np.ndarray<uint8> (h, w)
    centers: np.ndarray<float> (n_instances, emb_dim)
        centers of the current frame
    prev_centers: np.ndarray<float> (n_instances_prev, emb_dim)
        centers of the previous frame
    """
    # Skip if first element
    if prev_inst_seg is None:
        return inst_seg, available_keys, dict_ids

    # Update available keys
    available_keys = available_keys.difference(np.unique(prev_inst_seg))

    # Only background
    if len(np.unique(inst_seg)) == 1:
        return inst_seg, available_keys, dict_ids

    # Compute cost matrix: 1 - IoU for each instance in frame 1 and frame2
    if not use_barycenter:
        cost_matrix = compute_IoU_instances(prev_inst_seg, inst_seg)
    else:
        cost_matrix = euclidean_distances(prev_centers, centers)
    # Apply step 1 and step 3
    inst_seg, dict_ids = sync_ids(inst_seg, cost_matrix, dict_ids, available_keys, cost_threshold)

    if len(available_keys) == 0:
        # Reset keys, since enough timeframes separate the ids, we can reuse without any chance of overlapping
        print('Reset instance id keys.')
        available_keys = set(range(1, MAX_INSTANCES_SCENE))

    return inst_seg, available_keys, dict_ids


def make_ids_consecutives(x):
    unique_ids = np.unique(x)
    if unique_ids[0] == 0:
        dict_ids = dict(zip(unique_ids, np.arange(len(unique_ids))))
    else: # no background
        dict_ids = dict(zip(unique_ids, np.arange(1, len(unique_ids) + 1)))

    return np.vectorize(dict_ids.__getitem__)(x).astype(np.uint8)


def sync_ids(current_ids, cost_matrix, old_dict_ids=None, available_keys=None, cost_threshold=0.99):
    """ Synchronise ids with the previous ids.

    Parameters
    ----------
        current_ids: np.ndarray (N_CLASSES, height, width)
        cost_matrix: np.ndarray (n_instances_prev, n_instances_current)
        old_dict_ids: dict
            keys mapping previous frame instance ids to their original id
        available_keys: set
            available keys (256 max)
    """
    assert cost_matrix.ndim == 2, 'Cost matrix is not two dimensional: {}'.format(cost_matrix)
    new_ids_to_old_ids = {}

    # Step 1: for each instance, try to map to another in the previous frame
    dict_existing_to_old, assigned_cost = hungarian_algo(cost_matrix, old_dict_ids)
    for key, value in dict_existing_to_old.items():
        # If there is atleast some overlap
        if assigned_cost[key] < cost_threshold:
            new_ids_to_old_ids[key] = value

    # Step 3: assign remaining instances to a new unique id
    for j in range(cost_matrix.shape[1]):
        new_id = j + 1
        if new_id not in new_ids_to_old_ids:
            new_ids_to_old_ids[new_id] = available_keys.pop()

    new_frame = np.vectorize(new_ids_to_old_ids.__getitem__, otypes=[np.uint8])(current_ids)
    return new_frame, new_ids_to_old_ids


def hungarian_algo(cost_matrix, old_dict_ids=None):
    """ Compute the optimal assignment given a cost matrix.

    Parameters
    ----------
        cost_matrix: np.ndarray (n_inst, new_n_inst)
            cost matrix for the hungarian algorithm. n_inst and new_n_inst need not be equal.

    Returns
    -------
        dict_new_to_old: mapping from new ids to old ids.
        assigned_cost: mapping from new ids to cost
    """
    ids, new_ids = linear_sum_assignment(cost_matrix)
    assigned_cost = dict(zip(new_ids + 1, cost_matrix[ids, new_ids]))
    assigned_cost[0] = 0  #  background
    # need to synchronise with the ids with the original ids (the first that appeared)
    if old_dict_ids is not None and len(ids) > 0:
        ids = np.vectorize(old_dict_ids.__getitem__)(ids + 1) - 1

    # add one to indices, to account for background (index 0)
    dict_new_to_old = dict(zip(new_ids + 1, ids + 1))
    # Background id does not change
    dict_new_to_old[0] = 0
    return dict_new_to_old, assigned_cost


def compute_IoU_instances(frame1, frame2):
    """
    Parameters
    ----------
        frame1: np.ndarray (N_CLASSES, height, width)
            instance ids taking values from 0 (background) to n_instances1 included
        frame2: np.ndarray (N_CLASSES, height, width)
            instance ids taking values from 0 (background) to n_instances2 included

    Returns
    -------
        cost_matrix: np.ndarray (n_instances1, n_instances2)
            dissimilarity matrix between each instance in frame1 and frame2, based on IoU (background removed)
    """
    # Compute IoU matrix
    unique_id_frame1 = np.unique(frame1)
    unique_id_frame2 = np.unique(frame2)
    assert np.all(unique_id_frame1== np.arange(len(unique_id_frame1))) and \
           np.all(unique_id_frame2 == np.arange(len(unique_id_frame2)))

    cm = confusion_matrix(frame1.ravel(), frame2.ravel())
    normalising_array = np.ones_like(cm)
    # row
    normalising_array += cm.sum(axis=0).reshape((1, -1))
    # column
    normalising_array += cm.sum(axis=1).reshape((-1, 1))
    # substract array to remove values appearing twice
    normalising_array -= cm

    row_indices = np.unique(frame1)[1:]  # remove background
    col_indices = np.unique(frame2)[1:]
    # Compute IOU, amd remove row + colum related to background
    cm = (cm / normalising_array)[row_indices[:, None], col_indices].reshape((len(row_indices), len(col_indices)))

    cost_matrix = 1 - cm
    return cost_matrix


def increment_life_clusters(dict_centers, cluster_mean_life):
    for existing_id, (life, existing_center) in dict_centers.items():
        #  If too old, delete
        if life + 1 == cluster_mean_life:
            print('Delete id {}'.format(existing_id))
            dict_centers.pop(existing_id)
        dict_centers[existing_id] = (life + 1, existing_center)


def enforce_consistency_centers(inst_seg, centers, dict_centers, available_keys, cost_threshold=1.5,
                                cluster_mean_life=10, verbose=False):
    """
    Parameters
    ----------
    inst_seg: np.ndarray<uint8> (N_CLASSES, h, w)
    centers: np.ndarray<float> (n_instances, emb_dim)
        centers of the current frame
    dict_centers : dict (id) -> (life, mean)
        existing_id: int
            unique id of the instance
        life: int
            from 1 to cluster_mean_life
        existing_center: np.ndarray<float32> (emb_dim,)
    """
    # mapping from current_id to first_appeared_id
    id_mapping = {}
    #  Instance ids start at 1
    unique_ids = np.unique(inst_seg)
    assert 0 not in unique_ids, 'instance ids must start at 1'
    assert len(unique_ids) == len(centers), '{} unique ids for {} centers'.format(len(unique_ids), len(centers))
    dict_id_to_center = dict(zip(unique_ids, centers))
    remaining_ids = set(unique_ids)

    #  Initialise dict_centers
    if len(dict_centers) == 0:
        for id in remaining_ids:
            id_mapping[id] = available_keys.pop()
            # Life starts at 1
            dict_centers[id_mapping[id]] = (1, dict_id_to_center[id])

        inst_seg = np.vectorize(id_mapping.__getitem__, otypes=[np.uint8])(inst_seg)
        return inst_seg, available_keys, dict_centers

    to_remove = []
    prev_centers = []
    map_prev_hunga_id_to_existing_id = {}
    j = 0
    for existing_id, (life, existing_center) in dict_centers.items():
        if life == 1:
            prev_centers.append(existing_center)
            map_prev_hunga_id_to_existing_id[j] = existing_id

            j += 1

    if len(prev_centers) > 0:
        prev_centers = np.stack(prev_centers, axis=0)
        cost_matrix = euclidean_distances(prev_centers, centers)
        prev_hunga_ids, hunga_ids = linear_sum_assignment(cost_matrix)
        for i in range(len(prev_hunga_ids)):
            if cost_matrix[prev_hunga_ids[i], hunga_ids[i]] < cost_threshold:
                existing_id = map_prev_hunga_id_to_existing_id[prev_hunga_ids[i]]
                if verbose:
                    print('Considered id: {}'.format(hunga_ids[i] + 1))
                    print('Correspondence found with existing id {} and cost: {:.2f}'.format(
                        existing_id, cost_matrix[prev_hunga_ids[i], hunga_ids[i]]))
                # Update mapping
                id_mapping[hunga_ids[i] + 1] = existing_id
                # Update mean
                dict_centers[existing_id] = (0, dict_id_to_center[hunga_ids[i] + 1])
                to_remove.append(hunga_ids[i] + 1)
            else:
                if verbose:
                    print('Cost too high: {:.2f}'.format(cost_matrix[prev_hunga_ids[i], hunga_ids[i]]))

    remaining_ids = remaining_ids.difference(to_remove)

    to_remove = []
    for id in remaining_ids:
        if verbose:
            print('----------')
            print('Remaining considered id {}'.format(id))
        for existing_id, (life, existing_center) in dict_centers.items():
            best_cost = float('inf')
            best_existing_id = None
            if life > 1:
                cost = np.linalg.norm(dict_id_to_center[id] - existing_center, ord=2)
                if verbose:
                    print('Existing id {}, distance: {:.3f}'.format(existing_id, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_existing_id = existing_id
            if best_cost < cost_threshold:
                # Assign to existing id
                id_mapping[id] = best_existing_id
                # Update mean
                dict_centers[best_existing_id] = (0, dict_id_to_center[id])
                # Update remaining ids
                to_remove.append(id)

    remaining_ids = remaining_ids.difference(to_remove)

    #  Remaining ids get assigned a new unique id
    for id in remaining_ids:
        id_mapping[id] = available_keys.pop()
        dict_centers[id_mapping[id]] = (0, dict_id_to_center[id])

    #  Increase life by one
    increment_life_clusters(dict_centers, cluster_mean_life)

    # Map to tracked id
    inst_seg = np.vectorize(id_mapping.__getitem__, otypes=[np.uint8])(inst_seg)

    if len(available_keys) < 50:
        #  Reset keys, since enough timeframes separate the ids, we can reuse without any chance of overlapping
        print('Reset instance id keys.')
        available_keys = set(range(1, MAX_INSTANCES_SCENE))

    return inst_seg, available_keys, dict_centers
