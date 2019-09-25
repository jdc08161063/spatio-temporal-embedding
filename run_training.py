from options import TemporalModelOptions
from trainer import Trainer


if __name__ == '__main__':
    options = TemporalModelOptions()
    opt = options.parse()

    trainer = Trainer(opt)
    trainer.train()
