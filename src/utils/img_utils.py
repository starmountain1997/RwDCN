import os
from typing import List

import numpy as np
from PIL import Image


def save_batch_img(img_save_dir, x, prefix: str, key: List[str], psnr: float, ssim: float):
    x = np.transpose(x, (0, 2, 3, 1))
    b, _, _, _ = x.shape
    if len(key) != b:
        raise ValueError('The length of key must equal to batch_size!')
    for i in range(b):
        single_img = Image.fromarray(x[i])
        single_img.save(os.path.join(img_save_dir, f'{prefix}-{key[i]}-{psnr:.2f}-{ssim:.2f}.png'))
