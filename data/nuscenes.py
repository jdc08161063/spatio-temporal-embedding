import sys

import os
import torch

import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torchvision.transforms as transforms

from PIL import Image
from scipy.optimize import linear_sum_assignment
from glob import glob

from demo.predictor import COCODemo
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
import monodepth2.networks
from monodepth2.layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist

from common.utils import get_iou
from common.constants import ID_FILTER, N_CLASSES, MAX_INSTANCES, COST_THRESHOLD, NUSCENES_ROOT, SENSOR


def create_nuscenes_dataset(version, output_path, mode='train'):
    """
    For each video, we store a sequence of data with the following information:
        - image (H, W, 3) jpg
        - instance segmentation (H, W) np.array<np.uint8>
          with values from [0, MAX_INSTANCES-1]. Note that the instances are not aligned with position/velocity
          i.e. id 1 in instance segmentation corresponds to element 0 in position/velocity
          when including more classes, we can store a 4D tensor (N_CLASSES, H, W)
        - position (MAX_INSTANCES, 3) np.array
        - velocity (MAX_INSTANCES, 3) np.array

    """
    ## Yaw angle different on camera: https://github.com/nutonomy/nuscenes-devkit/issues/21
    # Load Mask R-CNN
    # update the config options with the config file
    cfg.merge_from_file(MASK_RCNN_CONFIG_FILE)
    # manual override some options
    # cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])

    mask_rcnn = COCODemo(
        cfg,
        confidence_threshold=0.8,
    )

    # Load NuScenes
    nusc = NuScenes(version=version, dataroot=NUSCENES_ROOT, verbose=True)
    scene_splits = create_splits_scenes()

    print('Begin iterating over Nuscenes')
    print('-' * 30)
    # Loop over dataset
    for scene in nusc.scene:
        # Ensure the scene belongs to the split
        if scene['name'] not in scene_splits[mode]:
            continue

        scene_path = os.path.join(output_path, mode, scene['name'])
        print('scene_path: {}'.format(scene_path))
        os.makedirs(scene_path, exist_ok=True)

        t = 0
        sample_token = scene['first_sample_token']
        while sample_token:
            print('Image {}'.format(t))
            sample = nusc.get('sample', sample_token)
            data = match_instance_seg_and_bbox(nusc, mask_rcnn, sample)

            if data is not None:
                data['image'].save(os.path.join(scene_path, '{:04d}_image_tmp.jpg'.format(t)))
                np.save(os.path.join(scene_path, '{:04d}_instance_seg_tmp.npy'.format(t)), data['instance_seg'])
                np.save(os.path.join(scene_path, '{:04d}_position_tmp.npy'.format(t)), data['position'])
                np.save(os.path.join(scene_path, '{:04d}_velocity_tmp.npy'.format(t)), data['velocity'])
                np.save(os.path.join(scene_path, '{:04d}_orientation_tmp.npy'.format(t)), data['orientation'])
                np.save(os.path.join(scene_path, '{:04d}_size_tmp.npy'.format(t)), data['size'])
                np.save(os.path.join(scene_path, '{:04d}_token_tmp.npy'.format(t)), data['token'])
                np.save(os.path.join(scene_path, '{:04d}_intrinsics_tmp.npy'.format(t)), data['intrinsics'])
                np.save(os.path.join(scene_path, '{:04d}_sample_token_tmp.npy'.format(t)), np.array([sample_token]))


            sample_token = sample['next']
            t += 1

        link_instance_ids(nusc, scene_path)
        print('------------------\n')

    print('Computing depth maps')
    print('-' * 30)
    # Compute depth map here.
    generate_depth(output_path, mode)
    print('Dataset saved.')


def match_instance_seg_and_bbox(nusc, mask_rcnn, sample):
    data_token = sample['data'][SENSOR]
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(data_token)

    label_boxes = []
    label_position = []
    label_velocity = []
    label_orientation = []
    label_size = []
    label_token = []
    for box in boxes:
        # TODO: generalise to pedestrians, cyclists
        if box.name == 'vehicle.car':
            # Convert 3d bounding box corners to image space
            corners = view_points(box.corners(), camera_intrinsic, True)[:2]
            x_min, y_min = corners[0].min(), corners[1].min()
            x_max, y_max = corners[0].max(), corners[1].max()

            bounding_box = {'x1': x_min, 'x2': x_max, 'y1': y_min, 'y2': y_max}
            label_boxes.append(bounding_box)
            label_position.append(box.center)
            label_velocity.append(box.velocity)
            label_orientation.append(box.orientation.elements)
            label_size.append(box.wlh)
            label_token.append(box.token)

    label_position = np.array(label_position, dtype=np.float32)
    label_velocity = np.array(label_velocity, dtype=np.float32)
    label_orientation = np.array(label_orientation, dtype=np.float32)
    label_size = np.array(label_size, dtype=np.float32)
    label_token = np.array(label_token)

    # Load image and compute Mask R-CNN instance segmentation
    pil_img = Image.open(data_path)
    img = np.array(pil_img)
    # Mask R-CNN takes BGR image as input
    predictions = mask_rcnn.compute_prediction(img[..., ::-1])
    top_predictions = mask_rcnn.select_top_predictions(predictions)
    filter_mask = top_predictions.get_field('labels') == ID_FILTER
    instance_seg = top_predictions.get_field('mask')[filter_mask].numpy().astype(np.bool)
    pred_boxes = top_predictions.bbox[filter_mask].numpy()

    if len(pred_boxes) > len(label_boxes):
        return None

    # Assign instance segmentation prediction to 3D bounding boxes
    cost_matrix = np.zeros((len(pred_boxes), len(label_boxes)))
    for i in range(len(pred_boxes)):
        for j in range(len(label_boxes)):
            pred_box = pred_boxes[i]
            top_left, bottom_right = pred_box[:2].tolist(), pred_box[2:].tolist()
            pred_box = {'x1': top_left[0], 'y1': top_left[1], 'x2': bottom_right[0], 'y2': bottom_right[1]}
            cost_matrix[i, j] = 1 - get_iou(pred_box, label_boxes[j])

    optimal_row, optimal_col = linear_sum_assignment(cost_matrix)
    if not np.all(optimal_row == np.arange(len(pred_boxes))):
        return None

    # Filter elements that got assigned a too high cost
    cost_filter = cost_matrix[optimal_row, optimal_col] < COST_THRESHOLD
    optimal_col = optimal_col[cost_filter]
    optimal_row = optimal_row[cost_filter]

    if len(optimal_row) > 0:
        instance_seg = instance_seg[optimal_row]
        label_position = label_position[optimal_col]
        label_velocity = label_velocity[optimal_col]
        label_orientation = label_orientation[optimal_col]
        label_size = label_size[optimal_col]
        label_token = label_token[optimal_col]
    else:
        instance_seg = np.empty(shape=(0, N_CLASSES, *img.shape[:2]))
        label_position = np.empty(shape=(0, 3))
        label_velocity = np.empty(shape=(0, 3))
        label_orientation = np.empty(shape=(0, 4))
        label_size = np.empty(shape=(0, 3))
        label_token = np.empty(shape=(0,))

    data = {'image': pil_img,
            'instance_seg': instance_seg,
            'position': label_position,
            'velocity': label_velocity,
            'orientation': label_orientation,
            'size': label_size,
            'token': label_token,
            'intrinsics': camera_intrinsic}
    return data


def link_instance_ids(nusc, scene_path):
    """ Link instance ids in the video."""
    print('Linking instance ids...')
    filenames = sorted(glob(os.path.join(scene_path, '*image_tmp.jpg')))

    dict_box_to_instance_token = {}
    for filename in filenames:
        token = np.load(filename[:-len('image_tmp.jpg')] + 'token_tmp.npy')
        for tok in token:
            dict_box_to_instance_token[tok] = nusc.get('sample_annotation', tok)['instance_token']

    # Instance token to unique id
    instance_tokens = np.unique(np.array(list(dict_box_to_instance_token.values())))
    dict_instance_token_to_id = dict(zip(instance_tokens, range(len(instance_tokens))))

    dict_box_to_id = {k: dict_instance_token_to_id[v] for k, v in dict_box_to_instance_token.items()}

    old_filelist = glob(os.path.join(scene_path, '*'))  # Store to remove at the end

    # Saving the files one more time ensures that the frame order is monotonically increasing by 1
    count = 0
    for filename in filenames:
        image = Image.open(filename)
        instance_seg = np.load(filename[:-len('image_tmp.jpg')] + 'instance_seg_tmp.npy')
        position = np.load(filename[:-len('image_tmp.jpg')] + 'position_tmp.npy')
        velocity = np.load(filename[:-len('image_tmp.jpg')] + 'velocity_tmp.npy')
        orientation = np.load(filename[:-len('image_tmp.jpg')] + 'orientation_tmp.npy')
        size = np.load(filename[:-len('image_tmp.jpg')] + 'size_tmp.npy')
        token = np.load(filename[:-len('image_tmp.jpg')] + 'token_tmp.npy')
        intrinsics = np.load(filename[:-len('image_tmp.jpg')] + 'intrinsics_tmp.npy')
        sample_token = np.load(filename[:-len('image_tmp.jpg')] + 'sample_token_tmp.npy')

        # TODO: handle cases where there are more than MAX_INSTANCES detections
        if len(instance_seg) > MAX_INSTANCES:
            continue

        h, w = instance_seg.shape[-2:]
        new_instance_seg = np.zeros((MAX_INSTANCES, N_CLASSES, h, w), dtype=np.bool)
        new_position = np.zeros((MAX_INSTANCES, 3), dtype=np.float32)
        new_velocity = np.zeros((MAX_INSTANCES, 3), dtype=np.float32)
        new_orientation = np.zeros((MAX_INSTANCES, 4), dtype=np.float32)
        new_size = np.zeros((MAX_INSTANCES, 3), dtype=np.float32)

        if len(token) > 0:
            token = np.vectorize(dict_box_to_id.__getitem__)(token)
            if token.max() >= (MAX_INSTANCES - 1):  # Remove 1 because we're concatenating an array of zeros to take argmax
                continue

            new_instance_seg[token] = instance_seg
            new_position[token] = position
            new_velocity[token] = velocity
            new_orientation[token] = orientation
            new_size[token] = size

        # Convert one-hot to np.uint8. Value 0 refer to background
        # Can convert back with:
        # onehot_instance_seg = np.eye(np.max(uint_instanc_seg) + 1)[uint_instance_seg][..., 1:]
        new_instance_seg = np.concatenate([np.zeros((1, N_CLASSES, *new_instance_seg.shape[2:])), new_instance_seg])
        new_instance_seg = new_instance_seg.argmax(axis=0).astype(np.uint8)
        image.save(os.path.join(scene_path, '{:04d}_image.jpg'.format(count)))
        np.save(os.path.join(scene_path, '{:04d}_instance_seg.npy'.format(count)), new_instance_seg)
        np.save(os.path.join(scene_path, '{:04d}_position.npy'.format(count)), new_position)
        np.save(os.path.join(scene_path, '{:04d}_velocity.npy'.format(count)), new_velocity)
        np.save(os.path.join(scene_path, '{:04d}_orientation.npy'.format(count)), new_orientation)
        np.save(os.path.join(scene_path, '{:04d}_size.npy'.format(count)), new_size)
        np.save(os.path.join(scene_path, '{:04d}_token.npy'.format(count)), token)
        np.save(os.path.join(scene_path, '{:04d}_intrinsics.npy'.format(count)), intrinsics)
        np.save(os.path.join(scene_path, '{:04d}_sample_token.npy'.format(count)), sample_token)

        count += 1

    # Remove old images
    for f in old_filelist:
        if 'tmp' in f:
            os.remove(f)
    print('Done!')


def generate_depth(output_path, mode, model_name='mono+stereo_1024x320', ext='jpg', no_cuda=False):
    # Load model
    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join('', model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = monodepth2.networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = monodepth2.networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    for image_path in glob(os.path.join(output_path, mode, '*')):
        print('Scene: {}'.format(image_path))
        # Searching folder for images
        paths = glob(os.path.join(image_path, '*.{}'.format(ext)))
        output_directory = image_path

        print("-> Predicting on {:d} test images".format(len(paths)))
        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for idx, image_path in enumerate(paths):

                if image_path.endswith("_disp.jpg"):
                    # don't try to predict disparity for a disparity image!
                    continue

                # Load image and preprocess
                input_image = pil.open(image_path).convert('RGB')
                original_width, original_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                # Saving numpy file
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                output_name = output_name[:-len('_image')]
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                scaled_disp, _ = disp_to_depth(disp_resized, 0.1, 100)
                scaled_disp = scaled_disp.view(original_height, original_width)
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)

                name_dest_im = os.path.join(output_directory, "{}_disp.jpg".format(output_name))
                im.save(name_dest_im)

                print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                    idx + 1, len(paths), name_dest_im))

    print('-> Done!')
