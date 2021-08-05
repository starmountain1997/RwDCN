from torch.utils.data import Dataset

from src.data.backend import Backend

cufed5_x4 = {
    "dataroot_lr": "/home/usrs/gzr1997/DS/CUFED5/lr",
    "dataroot_hr": "/home/usrs/gzr1997/DS/CUFED5/hr",
}


class CUFED5Dataset(Dataset):
    def __init__(self):
        super(CUFED5Dataset, self).__init__()
        self.backend = Backend(**cufed5_x4)

    def __len__(self):
        return len(self.backend.key)
