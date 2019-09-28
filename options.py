import argparse


class TemporalModelOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Temporal instance segmentation and depth options")

        # PATHS
        self.parser.add_argument('--config',
                                 type=str,
                                 default='',
                                 help='Path of the config file')

        self.parser.add_argument('--restore',
                                 type=str,
                                 default='',
                                 help='Path of the model to restore (weights, optimiser)')

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=6)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--seq_len",
                                 type=int,
                                 help="sequence length",
                                 default=5)

        # TRAINING options
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50])
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
