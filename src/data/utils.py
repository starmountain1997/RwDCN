import os
from collections import OrderedDict


def get_img_dict(root_dir: str) -> OrderedDict:
	# this function return a dict {img_key, img_save_path}
	# use the key, we can locate image on disk.
    clip_walker = os.walk(root_dir)
    img_dict = OrderedDict()
    for _, clip_nos, _ in clip_walker:
        for clip_no in sorted(clip_nos):
            frame_walker = os.walk(os.path.join(root_dir, clip_no))
            for full_path, _, frame_nos in frame_walker:
                for frame_no in sorted(frame_nos):
                    img_dict[f'{clip_no}_{frame_no[:-4]}'] = os.path.join(full_path, frame_no)
        return img_dict
