from torch.utils.data import Dataset

from src.data.backend import Backend
from src.data.cufed5.base_dataset import cufed5_x4


class RefDataset(Dataset):
    def __init__(self, ref_level):
        super(RefDataset, self).__init__()
        self.backend = Backend(**cufed5_x4)
        if ref_level > 5 or ref_level < 1:
            raise ValueError(f'ref_level must be between 1 and 5, but got: {ref_level}')
        self.ref_level = ref_level

    def __len__(self):
        return len(self.backend.key)

    def __getitem__(self, idx: int):
        key = self.backend.key[idx]
        lr_img = self.backend.lr_get(key)
        gt_img = self.backend.hr_get(key)
        clip_no, _ = key.split('_')
        ref_key = clip_no + '_' + str(self.ref_level)
        ref_img = self.backend.hr_get(ref_key)
        return {
            'lr': lr_img,
            'gt': gt_img,
            'ref': ref_img,
            'key': key
        }
