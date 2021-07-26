from src.data.reds.base_dataset import REDSDataset


def _get_ref_key(frame_no):
    return str(int(frame_no) // 10 + 4).zfill(8)


class Ref10Dataset(REDSDataset):
    def __init__(self, **kwargs):
        super(Ref10Dataset, self).__init__(**kwargs)
        self.set_key()

    def __getitem__(self, idx: int):
        key = self.backend.key[idx]
        clip_no, frame_no = key.split('_')
        ref_key = f'{clip_no}_{_get_ref_key(frame_no)}'
        lr_img = self.backend.lr_get(key)
        gt_img = self.backend.hr_get(key)
        ref_img = self.backend.hr_get(ref_key)
        lr_img, gt_img, ref_img = self.crop_img(lr_img, gt_img, ref_img)

        return {
            'lr': lr_img,
            'gt': gt_img,
            'ref': ref_img,
            'key': key
        }

    def set_key(self):
        def _is_valid_lr_key(lr_idx):
            if lr_idx % 10 == 4:
                return False
            else:
                return True

        self.backend.key = [
            v for v in self.backend.key if _is_valid_lr_key(int(v.split('_')[1]))
        ]


if __name__ == '__main__':
    ds = Ref10Dataset(mode='train', lr_size=[90, 90])
    for idx, data in enumerate(ds):
        print('fuck')
