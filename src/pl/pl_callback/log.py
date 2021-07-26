import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.pl.pl_module.sr_module import SRModule
from src.utils.img_utils import save_batch_img
from src.utils.os_utils import create_dirs


class SaveResultCallback(Callback):
    def __init__(self, img_save_dir):
        super(SaveResultCallback, self).__init__()
        self.test_img_save_dir = os.path.join(img_save_dir, 'test')
        self.val_img_save_dir = os.path.join(img_save_dir, 'val')
        create_dirs([self.test_img_save_dir, self.val_img_save_dir])

    def on_test_batch_end(
            self, trainer: pl.Trainer, pl_module: SRModule, outputs: Any, batch: Any, batch_idx: int,
            dataloader_idx: int
    ) -> None:
        save_batch_img(self.test_img_save_dir, outputs['sr'], 'sr', outputs['key'], outputs['test_psnr'],
                       outputs['test_ssim'])

    def on_validation_batch_end(self, trainer, pl_module: SRModule, outputs: Any, batch: Any, batch_idx: int,
                                dataloader_idx: int):
        if outputs['sr'] is not None:
            key_list = [
                f"{trainer.global_step}-{v}" for v in outputs['key']
            ]
            save_batch_img(self.val_img_save_dir, outputs['sr'], 'sr', key_list, outputs['val_psnr'],
                           outputs['val_ssim'])
