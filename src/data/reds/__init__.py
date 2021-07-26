from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.reds.ref_dataset import Ref10Dataset
from src.data.reds.sisr_dataset import SISRDataset

datasets = {
    'Ref10': Ref10Dataset,
    'SISR': SISRDataset
}


class REDSDataModule(pl.LightningDataModule):
    def __init__(self, dataset_type, batch_size: int, lr_size: List = None):
        super(REDSDataModule, self).__init__()
        self.reds_train = datasets[dataset_type](mode='train', lr_size=lr_size)
        self.reds_val = datasets[dataset_type](mode='val', lr_size=lr_size)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.reds_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.reds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.reds_val, batch_size=1)
