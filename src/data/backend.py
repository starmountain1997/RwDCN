from PIL import Image
from torchvision import transforms

from src.data.utils import get_img_dict


class Backend(object):
    def __init__(self, dataroot_lr: str, dataroot_hr: str, scale_factor: int = 4):
        self.to_tensor = transforms.ToTensor()
        self.hr_env = get_img_dict(dataroot_hr)
        self.lr_env = get_img_dict(dataroot_lr)
        self.scale_factor = scale_factor
        self.key = [v for v in self.lr_env.keys()]
        self.lr_size = []

    def lr_get(self, key):
        lr_img = Image.open(self.lr_env[key])
        self.lr_size = [(x - (x % (4 * self.scale_factor))) for x in lr_img.size]
        lr_img = lr_img.resize(self.lr_size, Image.BICUBIC)
        lr_img = self.to_tensor(lr_img)
        return lr_img

    def hr_get(self, key):
        if self.lr_size is None:
            raise ValueError('please run lr_get first!!!')
        hr_img = Image.open(self.hr_env[key])
        hr_size = [x * self.scale_factor for x in self.lr_size]
        hr_img = hr_img.resize(hr_size, Image.BICUBIC)
        hr_img = self.to_tensor(hr_img)
        return hr_img

    # def lr_get(self, key):
    #     lr_img = Image.open(self.lr_env[key])
    #     lr_img = self.to_tensor(lr_img)
    #     return lr_img
    #
    # def hr_get(self, key):
    #     hr_img = Image.open(self.hr_env[key])
    #     hr_img = self.to_tensor(hr_img)
    #     return hr_img
