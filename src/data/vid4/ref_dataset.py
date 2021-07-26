import torchvision.transforms

from src.data.vid4.base_dataset import Vid4Dataset


class RefFullDataset(Vid4Dataset):
    def __init__(self, **kwargs):
        super(RefFullDataset, self).__init__(**kwargs)
        self.ref_key = {
            'calendar_Frame': 'calendar_Frame 020',
            'city_Frame': 'city_Frame 017',
            'foliage_Frame': 'foliage_Frame 24',
            'walk_Frame': 'walk_Frame 23'
        }

    def _set_key(self) -> None:
        self.backend.lr_keys = [
            v for v in self.backend.lr_keys if v not in self.ref_key.values()
        ]
        self.backend.hr_keys = [
            v for v in self.backend.hr_keys if v not in self.ref_key.values()
        ]

    def __getitem__(self, idx: int):
        key = self.backend.key[idx]
        clip_no, frame_no = key.split(' ')
        ref_key = self.ref_key[clip_no]
        lr_img = self.backend.lr_get(key)
        gt_img = self.backend.hr_get(key)
        ref_img = self.backend.hr_get(ref_key)
        _, lh, lw = lr_img.shape
        lh = lh - (lh % 16)
        lw = lw - (lw % 16)
        hh = lh * 4
        hw = lw * 4
        lr_img = torchvision.transforms.CenterCrop((lh, lw))(lr_img)
        gt_img = torchvision.transforms.CenterCrop((hh, hw))(gt_img)
        ref_img = torchvision.transforms.CenterCrop((hh, hw))(ref_img)
        # FIXME: 为了使用flownet的网络所作
        return {
            'lr': lr_img,
            'gt': gt_img,
            'ref': ref_img,
            'key': key
        }
