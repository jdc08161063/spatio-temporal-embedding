import torch
from kitti.kitti_dataset import get_kitti_mask_dataloaders
from cityscapes.cityscapes_trainer import SemanticTrainer


class KittiMaskTrainer(SemanticTrainer):
    def create_loss(self):
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 10]).float())

    def create_data(self):
        self.h = self.params['h']
        self.w = self.params['w']
        self.train_iterator, self.val_iterator = get_kitti_mask_dataloaders(batch_size=self.params['batch_size'])
