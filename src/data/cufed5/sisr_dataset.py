from torch.utils.data import Dataset

from src.data.backend import Backend
from src.data.cufed5.base_dataset import cufed5_x4


class SISRDataset(Dataset):
    def __init__(self):
        super(SISRDataset, self).__init__()
        self.backend = Backend(**cufed5_x4)

    def __len__(self):
        return len(self.backend.key)

    def __getitem__(self, idx: int):
        key = self.backend.key[idx]
        lr_img = self.backend.lr_get(key)
        gt_img = self.backend.hr_get(key)
        return {
            'lr': lr_img,
            'gt': gt_img,
            'key': key
        }
