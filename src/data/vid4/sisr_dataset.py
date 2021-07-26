import torchvision

from src.data.vid4.base_dataset import Vid4Dataset


class SISRDataset(Vid4Dataset):
    def __init__(self, **kwargs):
        super(SISRDataset, self).__init__(**kwargs)

    def __getitem__(self, idx: int):
        key = self.backend.key[idx]
        lr_img = self.backend.lr_get(key)
        gt_img = self.backend.hr_get(key)
        _, lh, lw = lr_img.shape
        lh = lh - (lh % 16)
        lw = lw - (lw % 16)
        hh = lh * 4
        hw = lw * 4
        lr_img = torchvision.transforms.CenterCrop((lh, lw))(lr_img)
        gt_img = torchvision.transforms.CenterCrop((hh, hw))(gt_img)
        return {
            'lr': lr_img,
            'gt': gt_img,
            'key': key
        }
