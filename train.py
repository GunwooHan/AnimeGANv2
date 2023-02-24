import argparse
import glob
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datasets import Img2ImgDataset
from models import PlAnimeGANv2

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_dir', type=str, default='datasets/arcane')

# 모델 관련 설정
parser.add_argument('--gpus', type=int, default=4)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--project', type=str, default='AnimeGanV2')
parser.add_argument('--name', type=str, default='animeganv2')

# 학습 관련 설정
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adv_weight', type=float, default=0.5)
parser.add_argument('--lpips_weight', type=float, default=1.0)
parser.add_argument('--recon_weight', type=float, default=1.0)
parser.add_argument('--gp_weight', type=float, default=0.00001)
args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    src_images = np.array(sorted(glob.glob(os.path.join(args.data_dir, 'src', '*'))))
    tgt_images = np.array(sorted(glob.glob(os.path.join(args.data_dir, 'tgt', '*'))))

    wandb_logger = WandbLogger(project=args.project, name=args.name)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="train/fid_score",
        filename='{train/fid_score:.2f}',
    )

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor(),
    ])

    model = PlAnimeGANv2(args)

    train_ds = Img2ImgDataset(src_images, tgt_images, transform)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True,
                                                   drop_last=True)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=args.gpus,
                         precision=args.precision,
                         max_epochs=args.epochs,
                         # log_every_n_steps=1,
                         strategy='ddp',
                         # num_sanity_val_steps=0,
                         # limit_train_batches=5,
                         # limit_val_batches=1,
                         logger=wandb_logger,
                         # callbacks=[checkpoint_callback],
                         )

    trainer.fit(model, train_dataloaders=train_dataloader)
    wandb.finish()
