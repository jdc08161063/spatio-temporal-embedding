import os
import sys
import datetime
import json
import yaml
import socket
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict, deque
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from time import time
from shutil import copyfile
from sklearn.cluster import MeanShift

import neural.resnet as resnet
import monodepth.networks as networks
from monodepth.layers import SSIM, transformation_from_parameters, BackprojectDepth, Project3D, disp_to_depth,\
    get_smooth_loss
from monodepth.utils import sec_to_hm_str, normalize_image
from neural.model import TemporalEncoder, InstanceDecoder, DepthEmbedding, ResnetEncoder, ResnetDecoder
from neural.losses import discriminative_loss_loopy, motion_loss, discriminative_loss_sequence_static, mask_loss
from neural.utils import require_grad
from data.dataset import MotionDataset
from data.utils import torch_img_to_numpy, denormalise
from data.consistency import temporally_align_sequence
from metrics.metrics import MotionMetrics
from common.utils import Logger, load_config
from common.constants import N_CLASSES, MAX_INSTANCES_SCENE, BANDWIDTH, CLUSTERING_COST_THRESHOLD, CLUSTER_MEAN_LIFE,\
    MIN_PIXEL_THRESHOLD
from neural.utils import cluster, cluster_causal, cluster_frame_by_frame, print_model_spec
from visualisation.utils import heatmap_image, image_to_tensor, plot_labels_on_image, plot_embedding_clustering
from data.consistency import enforce_consistency_centers, make_ids_consecutives, increment_life_clusters


class Trainer():
    def __init__(self, opt):
        if opt.config and opt.restore:
            raise ValueError('Cannot start a new session and restore a model at the same time.')

        self.opt = opt

        # New session
        if len(opt.config) > 0:
            config = load_config(opt.config)
            self.config = config
            self.session_name = self.create_session_name()
            self.tensorboard = SummaryWriter(self.session_name, comment=self.config['tag'], flush_secs=1)
            self.save_session()
        elif len(opt.restore) > 0:
            # TODO: restore logger as well
            config = load_config(os.path.join(opt.restore, 'config.yml'))
            config['restore_path'] = opt.restore
            self.config = config
            self.session_name = opt.restore
            purge_step = torch.load(self.config['checkpoint_name'])['global_step']
            self.tensorboard = SummaryWriter(self.session_name, comment=self.config['tag'], flush_secs=1,
                                             purge_step=purge_step)
        else:
            raise ValueError('Must specify --config or --restore')

        # Contains keys: 'temporal_encoder' and 'instance_decoder'
        self.model = OrderedDict()
        self.optimizer = None

        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.train_metrics = None
        self.val_metrics = None

        self._train_dataloader_iter = None

        self.global_step = 0
        self.epoch = 0
        self.best_score = 0

        self.device = torch.device('cuda') if self.config['gpu'] else torch.device('cpu')

        # Depth model
        # checking height and width are multiples of 32
        assert self.config['h'] % 32 == 0, "'height' must be a multiple of 32"
        assert self.config['h'] % 32 == 0, "'width' must be a multiple of 32"

        self.load_depth_inputs = self.config['depth_loss']
        self.depth_models = OrderedDict()
        self.parameters_to_train = []
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.config['h'] // (2 ** scale)
            w = self.config['w'] // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        print('Creating model..')
        self.create_model()
        print('\nCreating data..')
        self.create_data()

        if self.config['load_depth'] and self.config['freeze_posenet']:
            print('Freezing posenet weights')
            for key in self.depth_models.keys():
                if key != 'depth':
                    require_grad(self.depth_models[key], False)

        self.create_optimizer()
        self.create_metrics()

        # Add receptive_field in config
        self.config['receptive_field'] = self.model['temporal_encoder'].receptive_field

        # Restore weights, optimizer, global_step
        if 'restore_path' in self.config:
            self.load_checkpoint()
        else:
            if self.config['load_depth']:
                # Maybe load weights depth decoder + posenet from trained monodepth
                if len(self.config['monodepth_weights_path']) > 0:
                    print('Loading monodepth weights from {} with keys: {}'.
                          format(self.config['monodepth_weights_path'], self.depth_models.keys()))
                    for key in self.depth_models.keys():
                        if key != 'encoder':
                            weights = torch.load(os.path.join(self.config['monodepth_weights_path'], '{}.pth'.format(key)))
                            self.depth_models[key].load_state_dict(weights)
                    print('Also restore encoder')
                    weights = torch.load(os.path.join(self.config['monodepth_weights_path'], 'encoder.pth'))
                    # Do no use strict loading, as the state_dict contains the unnecessary keys `height`,
                    # `width` and `use_stero`
                    if not self.config['depth_embedding']:
                        self.model['temporal_encoder'].encoder.model.load_state_dict(weights, strict=False)
                    else:
                        self.depth_models['encoder'].load_state_dict(weights, strict=False)
            self.move_model_to_device()

        print("There are {:d} training items and {:d} validation items\n".format(
            len(self.train_dataset), len(self.val_dataset)))

        if self.config['generate_mask']:
            self.mask_generator = {}
            self.mask_generator['encoder'] = ResnetEncoder().to(self.device)
            self.mask_generator['decoder'] = ResnetDecoder(segmentation=True, n_classes=2).to(self.device)
            print_model_spec(self.mask_generator['encoder'], 'Mask encoder')
            print_model_spec(self.mask_generator['decoder'], 'Mask encoder')

            # Load weights
            if self.config['mask_model_path']:
                print('Loading mask model weights from {}'.format(self.config['mask_model_path']))
                checkpoint = torch.load(self.config['mask_model_path'])

                self.mask_generator['encoder'].load_state_dict(checkpoint['encoder'])
                self.mask_generator['decoder'].load_state_dict(checkpoint['decoder'])
            require_grad(self.mask_generator['encoder'], False)
            require_grad(self.mask_generator['decoder'], False)
            self.mask_generator['encoder'].eval()
            self.mask_generator['decoder'].eval()

    def create_model(self):
        # Create shared temporal decoder
        self.model['temporal_encoder'] = TemporalEncoder(self.config, self.device)

        # Instance segmentation decoder
        if self.config['instance_loss'] or self.config['mask_loss']:
            self.model['instance_decoder'] = InstanceDecoder(
                decoder_name=self.config['decoder_name'], emb_dim=self.config['emb_dim'],
                instance=self.config['instance_loss'], mask=self.config['mask_loss'],
                config=self.config)

        if self.config['depth_embedding']:
            self.model['depth_embedding'] = DepthEmbedding(config=self.config)

        # Depth model
        if self.config['load_depth']:
            if self.config['depth_embedding']:
                self.depth_models['encoder'] = resnet.ResnetEncoder()
            # Decoder
            if self.config['decoder_name'] == 'resnet':
                self.depth_models["depth"] = resnet.ResnetDecoder(depth=True, scales=self.opt.scales)
            else:
                raise ValueError('Not recognised decoder name: {}'.format(self.config['decoder_name']))
            print_model_spec(self.depth_models["depth"], 'Depth decoder')

            # PoseNet
            self.depth_models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            print_model_spec(self.depth_models["pose_encoder"], 'Pose encoder')

            self.depth_models["pose"] = networks.PoseDecoder(
                self.depth_models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            print_model_spec(self.depth_models["pose"], 'Pose decoder')

    def move_model_to_device(self):
        for key in self.model.keys():
            self.model[key].to(self.device)

        for key in self.depth_models.keys():
            self.depth_models[key].to(self.device)

    def create_data(self):
        train_dataset = MotionDataset(self.config['data_path'], dataset=self.config['dataset'], mode='train',
                                      seq_len=self.opt.seq_len, h=self.config['h'], w=self.config['w'],
                                      num_scales=self.num_scales, load_depth_inputs=self.load_depth_inputs,
                                      saved_numpy=self.config['saved_numpy'])
        val_dataset = MotionDataset(self.config['data_path'], dataset=self.config['dataset'], mode='val',
                                    seq_len=self.opt.seq_len, h=self.config['h'], w=self.config['w'],
                                    num_scales=self.num_scales, load_depth_inputs=self.load_depth_inputs,
                                    saved_numpy=self.config['saved_numpy'])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                           num_workers=max(0, self.config['n_workers'] - 2), drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, shuffle=False,
                                         num_workers=max(0, self.config['n_workers'] - 3), drop_last=True)

    def create_optimizer(self):
        for key in self.model.keys():
            self.parameters_to_train += list(self.model[key].parameters())

        for key in self.depth_models.keys():
            self.parameters_to_train += list(self.depth_models[key].parameters())

        params_with_grad = filter(lambda p: p.requires_grad, self.parameters_to_train)
        self.optimizer = torch.optim.Adam(params_with_grad, self.opt.learning_rate)

    def create_metrics(self):
        self.train_metrics = MotionMetrics(self.config, self.tensorboard)
        self.val_metrics = MotionMetrics(self.config, self.tensorboard)

    def forward_model(self, batch):
        output = {}
        receptive_field = self.config['receptive_field']

        # Temporal encoding
        temporal_features = self.model['temporal_encoder'](batch['img'])

        # Instance decoding
        if self.config['instance_loss'] or self.config['mask_loss']:
            output.update(self.model['instance_decoder'](temporal_features))

        if self.config['depth_embedding']:
            # Compute depth
            x = batch['img']
            b, seq_len, c, h, w = x.shape

            encoder_outputs = self.depth_models['encoder'](x.view(b * seq_len, c, h, w))
            depth_output = self.depth_models['depth'](encoder_outputs)[('disp', 0)]
            depth_output = depth_output.view(b, seq_len, *depth_output.shape[1:])
            output['depth'] = depth_output

            output.update(self.model['depth_embedding'](output))


        # Depth decoding
        if self.config['depth_loss']:
            depth_output = {}
            for scale in self.opt.scales:
                depth_output[('disp', scale)] = []
                depth_output[("depth", 0, scale)] = []
                for frame_id in self.opt.frame_ids[1:]:
                    depth_output[("sample", frame_id, scale)] = []
                    depth_output[("color", frame_id, scale)] = []
                    depth_output[("color_identity", frame_id, scale)] = []

            for frame_id in self.opt.frame_ids[1:]:
                depth_output[('axisangle', 0, frame_id)] = []
                depth_output[('translation', 0, frame_id)] = []
                depth_output[('cam_T_cam', 0, frame_id)] = []

            # Depth model forward pass
            seq_len = self.opt.seq_len

            # TODO: remove loops
            min_time_index = max(1, receptive_field - 1)
            for t in range(min_time_index, seq_len - 1):
                depth_decoder_inputs = [temporal_features[i][:, t] for i in range(len(temporal_features))]
                depth_output_t = self.depth_models["depth"](depth_decoder_inputs)
                depth_output_t.update(self.predict_poses(batch['input_depth'], t))

                self.generate_images_pred(batch['input_depth'], depth_output_t, t)

                for key, value in depth_output_t.items():
                    depth_output[key].append(value)

            # Add dummy values for first few frames and last frame
            dummy = {key: torch.zeros_like(value) for key, value in depth_output_t.items()}
            for key in depth_output.keys():
                for j in range(min_time_index):
                    depth_output[key].insert(0, dummy[key])
                depth_output[key].append(dummy[key])

            # Stack in time dimension
            for key, value in depth_output.items():
                depth_output[key] = torch.stack(value, dim=1)

            # Add masking placeholder (will be filled in loss)
            for scale in self.opt.scales:
                depth_output["identity_selection/{}".format(scale)] = \
                    torch.zeros_like(depth_output[('disp', 0)]).squeeze(2)

            output.update(depth_output)
        return output

    def predict_poses(self, inputs, t):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input
        pose_feats = {f_i: inputs["color_aug", f_i, 0][:, t] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.depth_models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = self.depth_models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def generate_images_pred(self, inputs, outputs, t):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.config['h'], self.config['w']], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"][:, t]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)][:, t])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)][:, t], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)][:, t],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)][:, t]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, t):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)][:, t]
            color = inputs[("color", 0, scale)][:, t]
            target = inputs[("color", 0, source_scale)][:, t]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)][:, t]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)][:, t]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale][:, t]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)][:, t] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["depth_loss_scale_{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["depth_loss"] = total_loss
        return losses

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.model.values():
            m.train()
        for m in self.depth_models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.model.values():
            m.eval()
        for m in self.depth_models.values():
            m.eval()

    def generate_mask_forward(self, batch, output):
        b, s, c, h, w = batch['img'].shape

        with torch.no_grad():
            x = self.mask_generator['encoder'](batch['img'].view(b * s, c, h, w))
            mask = self.mask_generator['decoder'](x)['segmentation']
            mask = torch.argmax(mask, dim=1).byte()
            output['mask'] = mask.view(b, s, h, w)

    def post_forward_model(self, batch, output):
        """ Mean shift clustering to get predicted instance segmentation."""
        if self.config['mask_loss']:
            #  output['mask'] is (N, T, 2, H, W), take the argmax
            output['mask'] = torch.argmax(output['mask_logits'], dim=2).byte()
        elif self.config['generate_mask']:
            self.generate_mask_forward(batch, output)
        else:
            output['mask'] = batch['instance_seg'].squeeze(2) > 0

        if self.config['instance_loss']:
            if self.config['loss_name'] == 'temporal':
                cluster_fn = cluster
            elif self.config['loss_name'] == 'temporal_causal':
                cluster_fn = partial(cluster_causal, accumulate_embeddings=False)
            elif self.config['loss_name'] == 'temporal_causal_accum':
                cluster_fn = partial(cluster_causal, accumulate_embeddings=True)
            elif self.config['loss_name'] == 'static':
                cluster_fn = cluster_frame_by_frame
            else:
                raise ValueError('Not recognised loss function {}'.format(self.config['loss_name']))

            n_clusters, pred_instance_seg, cluster_centers = cluster_fn(output['y'].detach().cpu().numpy(),
                                                                        output['mask'].detach().cpu().numpy(),
                                                                        receptive_field=self.config['receptive_field'])

            # Temporally align using IoU if static model
            if self.config['loss_name'] == 'static':
                pred_instance_seg = temporally_align_sequence(pred_instance_seg)
            output['instance_seg'] = torch.from_numpy(pred_instance_seg)
        return output

    def loss_function(self, batch, output):
        losses = {}
        if self.config['loss_name'] in ['temporal', 'temporal_causal', 'temporal_causal_accum']:
            loss_fn = discriminative_loss_loopy
        elif self.config['loss_name'] == 'static':
            loss_fn = discriminative_loss_sequence_static
        else:
            raise ValueError('Not recognised loss function {}'.format(self.config['loss_name']))

        if self.config['instance_loss']:
            losses.update(loss_fn(batch, output, self.config, self.device))

        # Mask loss
        if self.config['mask_loss']:
            losses.update(mask_loss(batch, output))
        if self.config['motion_loss']:
            losses.update(motion_loss(batch, output, self.device))

        if self.config['depth_loss']:
            depth_loss = {'depth_loss': torch.tensor(0.0).to(self.device)}
            for scale in self.opt.scales:
                depth_loss["depth_loss_scale_{}".format(scale)] = torch.tensor(0.0).to(self.device)

            seq_len = self.opt.seq_len
            assert self.config['receptive_field'] >= 1
            min_time_idx = max(1, self.config['receptive_field'] - 1)
            assert min_time_idx < seq_len - 1
            for t in range(min_time_idx, seq_len - 1):
                depth_loss_t = self.compute_losses(batch['input_depth'], output, t)
                for key, value in depth_loss_t.items():
                    depth_loss[key] += value

            for key in depth_loss.keys():
                depth_loss[key] /= (seq_len - 1 - min_time_idx)
            losses.update(depth_loss)

        weights = {'v_loss': self.config['weight_instance'],
                   'd_loss': self.config['weight_instance'],
                   'reg_loss': self.config['weight_instance'],
                   'position_loss': self.config['weight_motion'],
                   'velocity_loss': self.config['weight_motion'],
                   'depth_loss': self.config['weight_depth'],
                   'mask_loss': self.config['weight_mask'],
                   }

        total_loss = torch.tensor(0.0).to(self.device)
        for key, value in losses.items():
            if 'scale' not in key:
                total_loss += weights[key] * value
        losses['total_loss'] = total_loss

        return losses

    def preprocess_batch(self, batch):
        for key in batch.keys():
            if key != 'input_depth':
                batch[key] = batch[key].to(self.device)
            else:
                if self.load_depth_inputs:
                    for depth_key in batch[key].keys():
                        batch[key][depth_key] = batch[key][depth_key].to(self.device)

    def train_step(self):
        #######
        # Fetch data
        t0 = time()
        batch = self._get_next_batch()
        self.preprocess_batch(batch)
        fetch_time = time() - t0

        #######
        # Train model
        t1 = time()
        output = self.forward_model(batch)
        losses = self.loss_function(batch, output)

        self.optimizer.zero_grad()
        if losses['total_loss'].item() > 0:  # total_loss=0 means not a single instance in the batch
            losses['total_loss'].backward()
        self.optimizer.step()
        model_update_time = time() - t1

        #######
        # Print
        if self.global_step % self.config['print_iter'] == 0 or self.global_step in [10, 50, 100, 200]:
            self.log_time(losses, step_duration=(time() - t0),
                          fetch_time=fetch_time,
                          model_update_time=model_update_time)
            for name, value in losses.items():
                self.tensorboard.add_scalar('train/' + name, value.item(), self.global_step)

        #######
        # Visualise
        if self.global_step % self.config['vis_iter'] == 0:
            t2 = time()
            output = self.post_forward_model(batch, output)

            self.train_metrics.update(batch, output)
            self.visualise(batch, output, 'train')
            if self.config['print_step_time'] and self.global_step % self.config['print_iter'] == 0:
                print('Visualisation: {:.2f}s'.format(time() - t2))

    def log_time(self, losses, step_duration, fetch_time, model_update_time):
        """ Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / step_duration
        time_so_far = time() - self.start_time
        training_time_left = (
            self.config['n_iterations'] / self.global_step - 1.0) * time_so_far if self.global_step > 0 else 0
        print_string = "Epoch {:>3} | iteration {:>6}/{} | examples/s: {:3.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.global_step, self.config['n_iterations'], samples_per_sec,
                                  losses['total_loss'], sec_to_hm_str(time_so_far), sec_to_hm_str(training_time_left)))
        print('Fetch time: {:.2f}s, model update time: {:.2f}'.format(fetch_time, model_update_time))
        for key, val in losses.items():
            print('{}: {:.3f}'.format(key, val))
        print('\n')

    def train(self):
        self.set_train()

        self.start_time = time()
        while self.global_step < self.config['n_iterations']:
            self.global_step += 1
            if self.global_step % len(self.train_dataloader) == 0:
                self.epoch += 1
                print('Training\n')
            self.train_step()

            if self.global_step % self.config['val_iter'] == 0:
                print('-' * 100)
                print('Validation\n')
                score = self.test()

                if score > self.best_score or score == 0:
                    print('New best score: {:.3f} -> {:.3f}. Saving model.'.format(self.best_score, score))
                    self.best_score = score
                    self.save_checkpoint()
                print('-' * 100)

        self.tensorboard.close()

    def test_step(self, batch, val_losses, iteration):
        self.preprocess_batch(batch)

        output = self.forward_model(batch)
        losses = self.loss_function(batch, output)

        for k in losses.keys():
            val_losses[k] = val_losses.get(k, 0) + losses[k].item()

        output = self.post_forward_model(batch, output)
        self.val_metrics.update(batch, output)

        if iteration == 0:
            self.visualise(batch, output, 'val')

    def test(self):
        self.set_eval()
        val_losses = {}

        with torch.no_grad():
            # Visualise oracle situations
            print('Oracle visualisations')
            self.oracle()

            effective_val_size = 0
            for iteration, batch in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                # Loads of overlap, speeds up testing
                if iteration % self.opt.seq_len == 0:
                    effective_val_size += 1
                    self.test_step(batch, val_losses, iteration)

                if self.config['debug'] and iteration >= 10:
                    break

            for k in val_losses.keys():
                val_losses[k] /= max(1, effective_val_size)

            for key, val in val_losses.items():
                print('{}: {:.3f}'.format(key, val))
            print('\n')
            for name, value in val_losses.items():
                self.tensorboard.add_scalar('val/' + name, value, self.global_step)

        print('-' * 30)
        print('Training metrics')
        print('-' * 30)
        self.train_metrics.evaluate(self.global_step, 'train')
        print('-' * 30)
        print('Validation metrics')
        print('-' * 30)
        val_score = self.val_metrics.evaluate(self.global_step, 'val')

        self.set_train()
        return val_score

    def oracle_step(self, i, idx, oracle_length=30):
        print('val index: {}'.format(idx))
        batch_idx = 0
        time_idx = -2

        mode = 'oracle_{}'.format(i)
        video = []

        h, w = self.config['h'], self.config['w']
        pred_instance_seg = np.zeros((1, oracle_length, N_CLASSES, h, w), dtype=np.uint8)

        # Accumualate embeddings
        if self.config['loss_name'] == 'temporal_causal_accum':
            print('Accumulating embeddings')
            accumulate_embeddings = True
            accumulated_y = deque(maxlen=self.opt.seq_len)
        else:
            accumulate_embeddings = False

        available_keys = set(range(1, MAX_INSTANCES_SCENE))  # max of 256 instances
        dict_centers = {}
        for t in tqdm(range(oracle_length), total=oracle_length):
            if idx + t >= len(self.val_dataset):
                break
            # Need to load a whole batch because depth model only works with a fixed batch_size
            batch = self._get_next_batch()
            oracle_data = self.val_dataset[idx + t]
            for key in batch.keys():
                if key != 'input_depth':
                    batch[key][0] = oracle_data[key]
                else:
                    if self.load_depth_inputs:
                        for depth_key in batch[key].keys():
                            batch[key][depth_key][0] = oracle_data[key][depth_key]

            self.preprocess_batch(batch)

            output = self.forward_model(batch)
            # Because of the depth mask is computed in the loss atm
            _ = self.loss_function(batch, output)

            if t == 0:
                # One old visualisation step for depth TODO: change
                output_tmp = self.post_forward_model(batch, output)
                self.visualise(batch, output_tmp, 'oracle_{}'.format(i), max_vis=1)
                output_tmp = None

            if self.config['mask_loss']:
                #  output['mask'] is (N, T, 2, H, W), take the argmax
                output['mask'] = torch.argmax(output['mask_logits'], dim=2).byte()
            elif self.config['generate_mask']:
                self.generate_mask_forward(batch, output)
            else:
                output['mask'] = batch['instance_seg'].squeeze(2) > 0
            # For oracle data, only implement cluster center distance to assign ids
            if self.config['instance_loss']:
                #####
                # Predict instance segmentation

                y = output['y'][batch_idx, time_idx].detach().cpu().numpy()
                mask = output['mask'][batch_idx, time_idx].detach().cpu().numpy()
                y = np.transpose(y, (1, 2, 0))
                mask = mask.astype(np.bool)

                try:
                    ms = MeanShift(BANDWIDTH, bin_seeding=True)
                    if mask.sum() > 0:
                        if accumulate_embeddings:
                            accumulated_y.append(y[mask])
                            ms_inputs = np.concatenate(accumulated_y, axis=0)
                        else:
                            ms_inputs = y[mask]
                        ms.fit(ms_inputs)
                        ms_start_index = len(ms_inputs) - len(y[mask])
                        instance_seg_t = (ms.labels_ + 1)[ms_start_index:]

                        # Remove small elements
                        full_instance_seg_t = np.zeros_like(mask).astype(np.uint8)
                        full_instance_seg_t[mask] = instance_seg_t
                        valid_ids = [id for id in np.unique(full_instance_seg_t) if
                                     (full_instance_seg_t == id).sum() > MIN_PIXEL_THRESHOLD]
                        full_instance_seg_t[~np.isin(full_instance_seg_t, valid_ids)] = 0

                        mask = full_instance_seg_t > 0

                        if mask.sum() > 0:
                            instance_seg_t = full_instance_seg_t[mask]
                            # Ids must be consecutive
                            instance_seg_t = make_ids_consecutives(instance_seg_t)

                            if accumulate_embeddings:
                                centers = []
                                for id in np.unique(instance_seg_t):
                                    full_instance_seg_t = np.zeros_like(mask).astype(np.uint8)
                                    full_instance_seg_t[mask] = instance_seg_t
                                    mask_id = (full_instance_seg_t == id)
                                    centers.append(y[mask_id].mean(axis=0))
                                centers = np.stack(centers, axis=0)
                            else:
                                centers = ms.cluster_centers_

                            consistent_instance_seg_t, available_keys, dict_centers = enforce_consistency_centers(
                                instance_seg_t,
                                centers, dict_centers, available_keys, cost_threshold=CLUSTERING_COST_THRESHOLD,
                                cluster_mean_life=CLUSTER_MEAN_LIFE)

                            pred_instance_seg[0, t, 0][mask] = consistent_instance_seg_t
                        else:
                            increment_life_clusters(dict_centers, CLUSTER_MEAN_LIFE)
                except:
                    pass

            #####
            # Add to tensorboard
            img_t = torch_img_to_numpy(batch['img'])[batch_idx, time_idx]
            imgs_cat = image_to_tensor(img_t)

            if self.config['instance_loss']:
                # GT instance seg
                predicted_image = plot_labels_on_image(img_t, batch['instance_seg'][batch_idx, time_idx].detach().cpu().numpy())
                predicted_image = image_to_tensor(predicted_image)
                imgs_cat = torch.cat([imgs_cat, predicted_image], dim=2)

                # Pred instance seg
                labeled_image = plot_labels_on_image(img_t, pred_instance_seg[0, t])
                labeled_image = image_to_tensor(labeled_image)
                imgs_cat = torch.cat([imgs_cat, labeled_image], dim=2)

                # Plot embedding clustering
                embedding_clustering = plot_embedding_clustering(output['y'][batch_idx, time_idx].detach().cpu().numpy(),
                    pred_instance_seg[0, t], output['mask'][batch_idx, time_idx].detach().cpu().numpy(), self.config, dpi=100)
                embedding_clustering = image_to_tensor(embedding_clustering)
                imgs_cat = torch.cat([imgs_cat, embedding_clustering], dim=2)

            if self.config['depth_loss']:
                disp = normalize_image(output[("disp", 0)].detach().cpu()).squeeze(2)
                disp_t = image_to_tensor(heatmap_image(disp[batch_idx, time_idx].detach().cpu().numpy()))
                imgs_cat = torch.cat([imgs_cat, disp_t], dim=2)

            if self.config['mask_loss']:
                gt_mask = (batch['instance_seg'].squeeze(2) > 0)[batch_idx, time_idx].detach().cpu().numpy()
                pred_mask = torch.argmax(output['mask_logits'], dim=2).byte()[batch_idx, time_idx].detach().cpu().numpy()
                # Ground truth in green, predicted in red
                combined_mask = np.zeros_like(batch['img'][batch_idx, time_idx].detach().cpu().numpy()).astype(np.float32)
                combined_mask = combined_mask.transpose((1, 2, 0))
                combined_mask[:, :, 1] = gt_mask
                combined_mask[:, :, 0] = pred_mask
                imgs_cat = torch.cat([imgs_cat, image_to_tensor(combined_mask)], dim=2)

            video.append(imgs_cat)

            batch, output = None, None


        # Dim (N, oracle_length, 3, H, W)
        if len(video) > 0:
            # was empty when seq_len=15
            video = torch.stack(video, dim=0).unsqueeze(0)
            self.tensorboard.add_video('{}/instance_continuous'.format(mode), video,
                                       global_step=self.global_step, fps=5)

    def oracle(self):
        if self.config['dataset'] == 'nuscenes':
            return
        oracle_length = 50
        # Apollo validation set
        if self.config['dataset'] == 'apollo':
            if not self.config['debug']:
                oracle_idx = [2073, 2950, 770, 3150, 2131]
            else:
                oracle_idx = [10]
        elif self.config['dataset'] == 'kitti':
            oracle_idx = [642, 724, 884, 2898]

        for i, idx in enumerate(oracle_idx):
            self.oracle_step(i, idx, oracle_length)

    def visualise(self, batch, output, mode='train', max_vis=4):
        receptive_field = self.config['receptive_field']
        min_time_idx = max(1, receptive_field - 1)

        if self.config['depth_loss']:
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    self.tensorboard.add_video(
                        "{}/color_frame_{}_scale_{}".format(mode, frame_id, s),
                        denormalise(batch['input_depth'][("color", frame_id, s)].cpu()[:max_vis, min_time_idx:-1]),
                        self.global_step, fps=1)
                    if s == 0 and frame_id != 0:
                        self.tensorboard.add_video(
                            "{}/color_pred_frame_{}_scale_{}".format(mode, frame_id, s),
                            denormalise(output[("color", frame_id, s)].cpu()[:max_vis, min_time_idx:-1]),
                            self.global_step, fps=1)

                # Disparity
                disp = normalize_image(output[("disp", s)].detach().cpu()[:max_vis, min_time_idx:-1]).squeeze(2)
                disp_video = []
                for i in range(disp.size(0)):
                    disp_video.append(torch.stack([image_to_tensor(heatmap_image(disp[i, t].detach().cpu().numpy()))
                                                   for t in range(disp.size(1))],
                                                  dim=0))
                disp_video = torch.stack(disp_video, dim=0)
                self.tensorboard.add_video(
                    "{}/disp_scale_{}".format(mode, s), disp_video, self.global_step, fps=1)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        self.tensorboard.add_video(
                            "{}/predictive_mask_frame_{}_scale_{}".format(mode, frame_id, s),
                            output["predictive_mask"][("disp", s)][:max_vis, min_time_idx:-1].unsqueeze(2).detach().cpu(),
                            self.global_step, fps=1)

                elif not self.opt.disable_automasking:
                    self.tensorboard.add_video(
                        "{}/automask_scale_{}".format(mode, s),
                        output["identity_selection/{}".format(s)][:max_vis, min_time_idx:-1].unsqueeze(2).detach().cpu(),
                        self.global_step, fps=1)

        # Instance and mask visualisation
        batch_np = {}
        batch_np['img'] = torch_img_to_numpy(batch['img'])
        batch_keys = ['instance_seg']
        if self.config['motion_loss']:
            batch_keys += ['position', 'velocity', 'intrinsics']
        for key in batch_keys:
            batch_np[key] = batch[key].cpu().numpy()

        batch_size = batch['img'].size(0)
        seq_len = batch['img'].size(1)
        video = []

        if self.config['depth_loss']:
            disp = normalize_image(output[("disp", 0)].detach().cpu()).squeeze(2)

        for img_ind in range(min(max_vis, batch_size)):
            imgs_seq = []
            for t in range(seq_len):
                # RGB image
                rgb_img = image_to_tensor(batch_np['img'][img_ind, t])
                imgs_cat = rgb_img

                if self.config['instance_loss']:
                    if t >= (receptive_field - 1):
                        # GT instance seg
                        labeled_image = plot_labels_on_image(batch_np['img'][img_ind, t], batch_np['instance_seg'][img_ind, t],
                                                             None, None, None, dpi=100)
                        labeled_image = image_to_tensor(labeled_image)
                        imgs_cat = torch.cat([imgs_cat, labeled_image], dim=2)

                        # Pred instance seg
                        predicted_image = plot_labels_on_image(batch_np['img'][img_ind, t],
                                                               output['instance_seg'][img_ind, t].detach().cpu().numpy(),
                                                               None, None, None, dpi=100)
                        predicted_image = image_to_tensor(predicted_image)
                        imgs_cat = torch.cat([imgs_cat, predicted_image], dim=2)

                        # Plot embedding clustering
                        embedding_clustering = plot_embedding_clustering(output['y'][img_ind, t].detach().cpu().numpy(),
                            output['instance_seg'][img_ind, t].detach().cpu().numpy(),
                            output['mask'][img_ind, t].detach().cpu().numpy(), self.config, dpi=100)
                        embedding_clustering = image_to_tensor(embedding_clustering)
                        imgs_cat = torch.cat([imgs_cat, embedding_clustering], dim=2)
                    else:
                        zero_tensor = torch.zeros_like(rgb_img)
                        imgs_cat = torch.cat([imgs_cat, zero_tensor, zero_tensor, zero_tensor], dim=2)

                if self.config['depth_loss']:
                    if t >= min_time_idx and t < (seq_len - 1):
                        disp_t = image_to_tensor(heatmap_image(disp[img_ind, t].detach().cpu().numpy()))
                        imgs_cat = torch.cat([imgs_cat, disp_t], dim=2)
                    else:
                        zero_tensor = torch.zeros_like(rgb_img)
                        imgs_cat = torch.cat([imgs_cat, zero_tensor], dim=2)

                if self.config['mask_loss']:
                    if t >= (receptive_field - 1):
                        gt_mask = (batch_np['instance_seg'].squeeze(2) > 0)[img_ind, t]
                        pred_mask = torch.argmax(output['mask_logits'], dim=2).byte()[img_ind, t].detach().cpu().numpy()
                        # Ground truth in green, predicted in red
                        combined_mask = np.zeros_like(batch_np['img'][img_ind, t]).astype(np.float32)
                        combined_mask[:, :, 1] = gt_mask
                        combined_mask[:, :, 0] = pred_mask
                        imgs_cat = torch.cat([imgs_cat, image_to_tensor(combined_mask)], dim=2)
                    else:
                        zero_tensor = torch.zeros_like(rgb_img)
                        imgs_cat = torch.cat([imgs_cat, zero_tensor], dim=2)

                imgs_seq.append(imgs_cat)

            # Video shape (N, T, C, H, W)
            imgs_seq = torch.stack(imgs_seq, dim=0).unsqueeze(0)
            video.append(imgs_seq)
        video = torch.cat(video, dim=0)
        # Note that colors might flicker because of conversion to gif
        # see `make_video` https://github.com/lanpa/tensorboardX/blob/master/tensorboardX/summary.py
        self.tensorboard.add_video('{}/instance'.format(mode), video,
                                   global_step=self.global_step, fps=1)

    def _get_next_batch(self):  # from training dataset
        if self._train_dataloader_iter is None:
            self._train_dataloader_iter = iter(self.train_dataloader)
        batch = None
        while batch is None:
            try:
                batch = next(self._train_dataloader_iter)
            except StopIteration:
                self._train_dataloader_iter = iter(self.train_dataloader)
        return batch

    def save_checkpoint(self):
        checkpoint_name = self.config['checkpoint_name']
        checkpoint = {}

        checkpoint['temporal_encoder'] = self.model['temporal_encoder'].state_dict()

        if self.config['instance_loss'] or self.config['mask_loss']:
            checkpoint['instance_decoder'] = self.model['instance_decoder'].state_dict()

        if self.config['depth_embedding']:
            checkpoint['depth_embedding'] = self.model['depth_embedding'].state_dict()

        if self.config['depth_loss']:
            depth_model_dict = {}
            # Depth decoder
            depth_model_dict['depth'] = self.depth_models["depth"].state_dict()
            # PoseNet
            depth_model_dict['pose_encoder'] = self.depth_models["pose_encoder"].state_dict()
            depth_model_dict['pose'] = self.depth_models["pose"].state_dict()

            checkpoint['depth_model'] = depth_model_dict

        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['global_step'] = self.global_step
        checkpoint['epoch'] = self.epoch
        checkpoint['best_score'] = self.best_score

        torch.save(checkpoint, checkpoint_name)

        # Export tensorboard to json
        self.tensorboard.export_scalars_to_json(os.path.join(self.session_name, 'tensorboard_scalars.json'))
        print('Model saved to {}'.format(checkpoint_name))

    def load_checkpoint(self):
        #######
        # 1. Model must to restored and then moved to device (multi-gpu compatibility)
        # 2. Optimiser must be restored after the model has been moved to the correct device.
        #    (https://github.com/pytorch/pytorch/issues/2830#issuecomment-336183179)
        # 3. The order of the parameters given to the Adam optimizer must be consistent
        #    need to use OrderedDict (https://github.com/piergiaj/pytorch-i3d/issues/38)

        checkpoint_name = self.config['checkpoint_name']
        map_location = 'cuda' if self.config['gpu'] else 'cpu'
        checkpoint = torch.load(checkpoint_name, map_location=map_location)

        self.model['temporal_encoder'].load_state_dict(checkpoint['temporal_encoder'])

        if self.config['instance_loss'] or self.config['mask_loss']:
            self.model['instance_decoder'].load_state_dict(checkpoint['instance_decoder'])

        if self.config['depth_embedding']:
             self.model['depth_embedding'].load_state_dict(checkpoint['depth_embedding'])

        if self.config['depth_loss']:
            depth_model_dict = checkpoint['depth_model']
            # Depth decoder
            self.depth_models["depth"].load_state_dict(depth_model_dict['depth'])
            # PoseNet
            self.depth_models["pose_encoder"].load_state_dict(depth_model_dict['pose_encoder'])
            self.depth_models["pose"].load_state_dict(depth_model_dict['pose'])

        self.move_model_to_device()

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if 'best_score' in checkpoint:
            self.best_score = checkpoint['best_score']

        print('Loaded model and optimizer weights from {}'.format(checkpoint_name))

    def create_session_name(self):
        now = datetime.datetime.now()
        session_name = 'session_{}_{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}_{}'.format(socket.gethostname(),
                                                                                        now.year, now.month,
                                                                                        now.day, now.hour,
                                                                                        now.minute, now.second,
                                                                                        self.config['tag'])
        if self.opt.batch_size != 8:
            session_name += '_batch_{}'.format(self.opt.batch_size)
        if self.opt.seq_len != 5:
            session_name += '_seqlen_{}'.format(self.opt.seq_len)
        if self.opt.learning_rate != 0.0003:
            session_name += '_lr_{}'.format(self.opt.learning_rate)

        session_name = os.path.join(self.config['output_path'], session_name)
        return session_name

    def save_session(self):
        os.makedirs(self.session_name, exist_ok=True)
        self.config['session_name'] = self.session_name
        self.config['checkpoint_name'] = os.path.join(self.session_name, 'checkpoint-latest')
        # Save config here.
        self._save_config()
        # Save options
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(self.session_name, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

        # Save logs
        # stdout and stderr to be displayed both on the terminal and in an output file.
        logger = Logger(os.path.join(self.session_name, 'logs.txt'))
        sys.stdout = logger
        sys.stderr = logger

    def _save_config(self):
        # Exact copy of the yaml file as defined in the experiments folder
        copyfile(self.config['config_path'], os.path.join(self.config['session_name'], 'raw_config.yml'))
        with open(self.config['config_path']) as raw_config_yml:
            raw_config_yml_string = raw_config_yml.read()
            # Add batch_size, seq_len, learning_rate
            raw_config_yml_string += '\n\nbatch_size: {}\nseq_len: {}\nlearning_rate: {}'.format(self.opt.batch_size,
                                                                                                 self.opt.seq_len,
                                                                                                 self.opt.learning_rate)
            self.tensorboard.add_text('Config YAML', raw_config_yml_string.replace('\n', '  \n'), 0)
        # File with all the fields imported
        with open(os.path.join(self.config['session_name'], 'config.yml'), 'w') as f:
            yaml.safe_dump(self.config, f)