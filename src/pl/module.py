import random
import time
from abc import abstractmethod
from unittest.mock import patch

import numpy as np
import pytorch_lightning as pl
import torch
from mmedit.models import PerceptualLoss, PerceptualVGG, CharbonnierLoss, GANLoss
from pytorch_lightning.metrics.functional import psnr, ssim

from src.losses.texture_loss import TextureLoss


class SRModule(pl.LightningModule):
    def __init__(self,
                 lr=1e-4,
                 betas=[0.9, 0.999],
                 scheduler='CosineAnnealingLR',  # FIXME:
                 # milestones=[20000, 200000],
                 # gamma=0.5,
                 # factor=0.5,
                 # patience=4,
                 T_max=32,
                 l_carbonnier=1., l_percep=1e-4, l_style=1e-4, l_adv=1e-6, l_texture=1e-4):
        super(SRModule, self).__init__()
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, batch):
        pass

    @patch.object(PerceptualVGG, 'init_weights')
    def configure_loss(self, init_weights=1.):
        self.percep = PerceptualLoss(layer_weights={'4': 1., '9': 1., '18': 1.})
        self.carbonnier = CharbonnierLoss()
        self.gan = GANLoss('wgan', loss_weight=0.0001)
        self.texture = TextureLoss()

    def cal_loss(self, sr, gt):
        percep, style = self.percep(sr, gt)
        carbonnier = self.carbonnier(sr, gt)
        gan = self.gan(sr, True) + self.gan(gt, False)
        texture = self.texture(sr, gt)

        return carbonnier * self.hparams.l_carbonnier + percep * self.hparams.l_percep + style * self.hparams.l_style + gan * self.hparams.l_adv + texture * self.hparams.l_texture

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=0,
                                          betas=self.hparams.betas)
        if self.hparams.scheduler == 'CosineAnnealingLR':
            self.scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.hparams.T_max
                )
            }
        else:
            raise ValueError(f'invalid scheduler {self.hparams.scheduler}.')
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, idx):
        sr = self.forward(batch)
        train_loss = self.cal_loss(sr, batch['gt'])
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, idx):
        sr = self.forward(batch)
        gt = batch['gt']
        sr = torch.clamp(sr, 0., 1.)
        gt = torch.clamp(gt, 0., 1.)
        val_loss = self.cal_loss(sr, batch['gt'])
        val_psnr = psnr(sr, batch['gt'], data_range=1.)
        val_ssim = ssim(sr, batch['gt'], data_range=1.)
        self.log('val_loss', val_loss)
        self.log('val_psnr', val_psnr)
        self.log('val_ssim', val_ssim)
        if_save = random.randint(0, 100)
        if if_save == 42:
            sr = sr * 255.
            sr = sr.data.cpu().numpy().astype(np.uint8)
            return {'sr': sr,
                    'key': batch['key'],
                    'val_psnr': val_psnr,
                    'val_loss': val_loss,
                    'val_ssim': val_ssim}
        else:
            return {'sr': None,
                    'key': batch['key'],
                    'val_psnr': val_psnr,
                    'val_loss': val_loss,
                    'val_ssim': val_ssim}

    def test_step(self, batch, idx):
        starttime = time.time()
        sr = self.forward(batch)
        gt = batch['gt']
        sr = torch.clamp(sr, 0., 1.)
        gt = torch.clamp(gt, 0., 1.)
        endtime = time.time()
        memory_a = torch.cuda.memory_allocated(0)
        test_psnr = psnr(sr, gt, data_range=1.)
        test_ssim = ssim(sr, gt, data_range=1.)
        run_time = endtime - starttime
        self.log('test_psnr', test_psnr, on_step=True, on_epoch=True)
        self.log('test_ssim', test_ssim, on_step=True, on_epoch=True)
        self.log('run_time', run_time, on_step=True, on_epoch=True)
        self.log('memory allocated', memory_a, on_step=True, on_epoch=True)
        sr = sr * 255.
        sr = sr.data.cpu().numpy().astype(np.uint8)
        return {'sr': sr,
                'key': batch['key'],
                'test_psnr': test_psnr,
                'test_ssim': test_ssim}
