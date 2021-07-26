from torch.utils.data import Dataset

from src.data.backend import Backend

vid4_x4 = {
    "scale_factor": 4,
    "dataroot_lr": "/home/usrs/gzr1997/DS/Vid4/LR",
    "dataroot_hr": "/home/usrs/gzr1997/DS/Vid4/HR",
}


class Vid4Dataset(Dataset):
    def __init__(self):
        super(Vid4Dataset, self).__init__()
        self.backend = Backend(**vid4_x4)

    def __len__(self):
        return len(self.backend.key)
