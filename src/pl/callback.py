import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.pl.module import SRModule
from src.utils.img_utils import save_batch_img
from src.utils.os_utils import create_dirs

# this callback can save the top k checkpoint by SSIM.
class SSIMCkptCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, dirpath, save_top_k=5):
        super(SSIMCkptCallback, self).__init__(monitor='val_ssim', dirpath=dirpath,
                                               filename='{epoch:02d}-{val_ssim:.4f}', save_top_k=save_top_k, mode='max')

# this callback can save the top k checkpoint by Loss
class LossCkptCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, dirpath, save_top_k=5):
        super(LossCkptCallback, self).__init__(monitor='val_loss', dirpath=dirpath,
                                               filename='{epoch:02d}-{val_loss:.4f}', save_top_k=save_top_k, mode='min')


class SaveResultCallback(Callback):
	# this callback save image during training or testing.
	# notice:
	# every time one epoch ends, pythorch lightning will run test task. and the on_test_batch_end will be called after the test task ends.
	# you can ref: https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks
    def __init__(self, img_save_dir):
        super(SaveResultCallback, self).__init__()
        self.test_img_save_dir = os.path.join(img_save_dir, 'test')
        self.val_img_save_dir = os.path.join(img_save_dir, 'val')
        create_dirs([self.test_img_save_dir, self.val_img_save_dir])

    def on_test_batch_end(
            self, trainer: pl.Trainer, pl_module: SRModule, outputs: Any, batch: Any, batch_idx: int,
            dataloader_idx: int
    ) -> None:
	# save output during training.
        save_batch_img(self.test_img_save_dir, outputs['sr'], 'sr', outputs['key'], outputs['test_psnr'],
                       outputs['test_ssim'])

    def on_validation_batch_end(self, trainer, pl_module: SRModule, outputs: Any, batch: Any, batch_idx: int,
                                dataloader_idx: int):
		# save output during testing.
        if outputs['sr'] is not None:
            key_list = [
                f"{trainer.global_step}-{v}" for v in outputs['key']
            ]
            save_batch_img(self.val_img_save_dir, outputs['sr'], 'sr', key_list, outputs['val_psnr'],
                           outputs['val_ssim'])
