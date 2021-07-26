import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.vid4.ref_dataset import RefFullDataset
from src.data.vid4.sisr_dataset import SISRDataset

__all__ = [
    'Vid4DataModule'
]

datasets = {
    'SISR': SISRDataset,
    'Ref10': RefFullDataset
}


class Vid4DataModule(pl.LightningDataModule):
    def __init__(self, dataset_type: str):
        super(Vid4DataModule, self).__init__()
        self.vid4_test = datasets[dataset_type]()

    def test_dataloader(self):
        return DataLoader(self.vid4_test, batch_size=1)
