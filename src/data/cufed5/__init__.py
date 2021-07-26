import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.cufed5.ref_dataset import RefDataset
from src.data.cufed5.sisr_dataset import SISRDataset


class CUFED5DataModule(pl.LightningDataModule):
    def __init__(self, dataset_type: str):
        super(CUFED5DataModule, self).__init__()
        if dataset_type == 'Ref1':
            self.cufed5_test = RefDataset(ref_level=1)
        elif dataset_type == 'Ref2':
            self.cufed5_test = RefDataset(ref_level=2)
        elif dataset_type == 'Ref3':
            self.cufed5_test = RefDataset(ref_level=3)
        elif dataset_type == 'Ref4':
            self.cufed5_test = RefDataset(ref_level=4)
        elif dataset_type == 'Ref5':
            self.cufed5_test = RefDataset(ref_level=5)
        elif dataset_type == 'SISR':
            self.cufed5_test = SISRDataset()

    def test_dataloader(self):
        return DataLoader(self.cufed5_test, batch_size=1)
