from PIL import Image

from src.data.backend import Backend


class REDSBackend(Backend):
	#. REDS's backend.
	# this backend is 

	# where	
    def __init__(self, **kwargs):
        super(REDSBackend, self).__init__(**kwargs)

    def lr_get(self, key):
		# read low resolution image by key.
        lr_img = Image.open(self.lr_env[key])
        lr_img = self.to_tensor(lr_img)
        return lr_img

    def hr_get(self, key):
		# read high resolution image by. key.
        hr_img = Image.open(self.hr_env[key])
        hr_img = self.to_tensor(hr_img)
        return hr_img
