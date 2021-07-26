import numpy as np
from PIL import Image

from src.data.reds.base_dataset import REDSDataset


class SISRDataset(REDSDataset):
    def __init__(self, **kwargs):
        super(SISRDataset, self).__init__(**kwargs)

    def __getitem__(self, idx: int):
        key = self.backend.key[idx]
        lr_img = self.backend.lr_get(key)
        gt_img = self.backend.hr_get(key)
        lr_img, gt_img, ref_img = self.crop_img(lr_img, gt_img)
        return {
            'lr': lr_img,
            'gt': gt_img,
            'key': key
        }


if __name__ == '__main__':
    ds = SISRDataset(mode='train', lr_size=[90, 90])
    for idx, data in enumerate(ds):
        lr_img = data['lr']
        gt_img = data['gt']
        lr_img = lr_img * 255.
        gt_img = gt_img * 255.
        lr_img = lr_img.numpy().astype(np.uint8)
        gt_img = gt_img.numpy().astype(np.uint8)
        lr_img = np.transpose(lr_img, (1, 2, 0))
        gt_img = np.transpose(gt_img, (1, 2, 0))
        key = data['key']
        lr_img = Image.fromarray(lr_img)
        gt_img = Image.fromarray(gt_img)
        lr_img.save(f'/home/usrs/gzr1997/tmp/lr-{key}.png')
        gt_img.save(f'/home/usrs/gzr1997/tmp/gt-{key}.png')
        print(key)
