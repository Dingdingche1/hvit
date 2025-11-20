import argparse
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from src.data.datasets import get_dataloader
from src.trainer_diffusion import DiffusionRegistrationModule
from src.utils import read_yaml_file


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM-driven registration with mixed precision")
    parser.add_argument('--config', type=str, default='config/diffusion_train.yaml', help='Path to training YAML config')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment', type=str, default=None, help='Optional experiment name override')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = read_yaml_file(config_path)

    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    optim_cfg = config.get('optimization', {})

    experiment_name = args.experiment or config.get('experiment', 'diffusion_registration')

    wandb_logger = WandbLogger(project='wandb_HViT', name=experiment_name)

    train_loader = get_dataloader(
        data_path=data_cfg.get('train_path'),
        input_dim=data_cfg.get('input_dim'),
        batch_size=data_cfg.get('batch_size', 2),
        shuffle=True,
        is_pair=False
    )
    val_loader = get_dataloader(
        data_path=data_cfg.get('val_path'),
        input_dim=data_cfg.get('input_dim'),
        batch_size=data_cfg.get('batch_size', 2),
        shuffle=False,
        is_pair=data_cfg.get('paired_validation', True)
    )

    module = DiffusionRegistrationModule(config)

    trainer = Trainer(
        max_epochs=training_cfg.get('max_epochs', 200),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=training_cfg.get('devices', 1),
        logger=[wandb_logger],
        precision=optim_cfg.get('trainer_precision', '32-true'),
        accumulate_grad_batches=optim_cfg.get('gradient_accumulation', 1),
        gradient_clip_val=optim_cfg.get('grad_clip', 0.0),
        log_every_n_steps=training_cfg.get('log_every_n_steps', 10)
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
    wandb_logger.experiment.finish()


if __name__ == '__main__':
    main()
