import argparse
import torch

from kitti.kitti_trainer import KittiMaskTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",
                    type=int,
                    help="batch size",
                    default=16)
parser.add_argument("--output_path",
                    type=str,
                    help='output path',
                    default='')
parser.add_argument("--model_name",
                    type=str,
                    help="model name",
                    default='resnet',
                    choices=['resnet', 'deeplab'])
parser.add_argument('--tag',
                    type=str,
                    help='session tag',
                    default='baseline')

options = parser.parse_args()

params = {'batch_size': options.batch_size,
          'output_dir': options.output_path,
          'tag': options.tag,
          'device': torch.device('cuda'),
          'model_name': options.model_name,
          'n_classes': 2,
          'pretrained_path': '',
          'h': 192,
          'w': 640
          }

trainer = KittiMaskTrainer(params)
trainer = trainer.to(params['device'])

trainer.train_model(n_epochs=5, save_every=100)