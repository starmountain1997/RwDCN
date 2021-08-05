from typing import List

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

from src.data.reds.backend import REDSBackend

# we. use REDS for training and validation.
# here defines where the training and validation data is saved.

reds_train = {
    "dataroot_lr": "/home/usrs/gzr1997/DS/REDS_DATA/train/sharp_bicubic", # low resolution path.
    "dataroot_hr": "/home/usrs/gzr1997/DS/REDS_DATA/train/sharp", # high resolution path.
}

reds_val = {
    "dataroot_lr": "/home/usrs/gzr1997/DS/REDS_DATA/val/sharp_bicubic",
    "dataroot_hr": "/home/usrs/gzr1997/DS/REDS_DATA/val/sharp",
}


class REDSDataset(Dataset):
    def __init__(self, mode, lr_size: List, scale_factor=4):
        super(REDSDataset, self).__init__()
        self.mode = mode # what mode they us, train or test?
        self.crop_indices = [] # use it in random crop.
        self.scale_factor = scale_factor
        self.lr_size = lr_size
        self.hr_size = [x * self.scale_factor for x in self.lr_size]
        if self.mode == 'train':
            self.backend = REDSBackend(**reds_train)
        elif self.mode == 'val':
            self.backend = REDSBackend(**reds_val)

    def crop_img(self, lr_img, gt_img, ref_img=None):
        if self.mode == 'train':
            self.crop_indices = transforms.RandomCrop.get_params(lr_img, output_size=self.lr_size)
            i, j, h, w = self.crop_indices
            lr_img = crop(lr_img, i, j, h, w)
            gt_img = crop(gt_img, i * 4, j * 4, h * 4, w * 4)
            if ref_img is not None:
                ref_img = crop(ref_img, i * 4, j * 4, h * 4, w * 4)
            return lr_img, gt_img, ref_img
        elif self.mode == 'val':
            lr_img = transforms.CenterCrop(self.lr_size)(lr_img)
            gt_img = transforms.CenterCrop(self.hr_size)(gt_img)
            if ref_img is not None:
                ref_img = transforms.CenterCrop(self.hr_size)(ref_img)
            return lr_img, gt_img, ref_img

    def __len__(self):
        return len(self.backend.key)
