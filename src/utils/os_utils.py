import os
from typing import List

import torch
from PIL import Image

from src.data.utils import get_img_dict


def create_dirs(path_list: List[str]):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)


def move_to_different_directory(dir):
    files = os.listdir(dir)
    for file in files:
        if not file.endswith('.png'):
            continue
        clip_no, frame_no = file.split('_')
        if not os.path.exists(os.path.join(dir, clip_no)):
            os.makedirs(os.path.join(dir, clip_no))
        print(frame_no)
        os.rename(os.path.join(dir, file),
                  os.path.join(dir, clip_no, frame_no))


def generate_x4_lr_data(dir):
    hr_keys = get_img_dict(dir)
    for key, value in hr_keys.items():
        if os.path.basename(value) == '0.png':
            img = Image.open(value)
            img = img.resize((x // 4 for x in img.size), Image.BICUBIC)
            lr_img_save_path = os.path.join('/home/usrs/gzr1997/DS/FFFF', key.split('_')[0])
            print(lr_img_save_path)
            if not os.path.exists(lr_img_save_path):
                os.makedirs(lr_img_save_path)
            img.save(os.path.join(lr_img_save_path, '0.png'))


def convert_srntt_model():
    state_dict = {
        'srntt.' + k: v for k, v in torch.load('/home/usrs/gzr1997/CODE/RwDCN/pretrained/netG_100.pth').items()
    }

    torch.save(
        {
            'state_dict': state_dict
        }, '/home/usrs/gzr1997/CODE/RwDCN/pretrained/wrap_srntt_netG_100.pth'
    )


def convert_crossnet_model():
    state_dict = {
        'crossnet.' + k: v for k, v in torch.load('/home/usrs/gzr1997/CODE/RwDCN/pretrained/CP185000.pth').items()
    }
    torch.save(
        {
            'state_dict': state_dict
        }, '/home/usrs/gzr1997/CODE/RwDCN/pretrained/wrap_crossnet.pth'
    )


def convert_rcan_model():
    state_dict = {
        'rcan.' + k: v for k, v in torch.load('/home/usrs/gzr1997/CODE/RwDCN/pretrained/RCAN_BIX4.pt').items()
    }
    torch.save(
        {
            'state_dict': state_dict
        }, '/home/usrs/gzr1997/CODE/RwDCN/pretrained/wrap_rcan.pth'
    )


def convert_rcan_model(dir):
    state_dict = torch.load(dir)
    state_dict = state_dict.pop('hyper_parameters', None)
    torch.save(state_dict, '/home/usrs/gzr1997/RES/RCAN_ImageSRREDS_x4_train/net/de_epoch=37-val_loss=0.03.ckpt')


def convert_rwdcn_model(dir):
    state_dict = torch.load(dir)
    fuck_me = state_dict['state_dict']
    torch.save(
        {'state_dict': fuck_me},
        '/home/usrs/gzr1997/CODE/RwDCN/pretrained/rwdcn_lr.pth'
    )
    print('fuck me')


if __name__ == '__main__':
    convert_rwdcn_model(
        '/home/usrs/gzr1997/CODE/RwDCN/pretrained/rwdcn_lr.pth')
