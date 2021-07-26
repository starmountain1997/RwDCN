# class CUFED5Backend(Backend):
#     def __init__(self, dataroot_hr='/home/usrs/gzr1997/DS/CUFED5/hr', dataroot_lr='/home/usrs/gzr1997/DS/CUFED5/lr'):
#         super(CUFED5Backend, self).__init__(dataroot_lr=dataroot_lr, dataroot_hr=dataroot_hr)
#         self.lr_size = []
#
#     def lr_get(self, key):
#         lr_img = Image.open(self.lr_env[key])
#         self.lr_size = [(x - (x % (4 * self.scale_factor))) for x in lr_img.size]
#         lr_img = lr_img.resize(self.lr_size, Image.BICUBIC)
#         lr_img = self.to_tensor(lr_img)
#         return lr_img
#
#     def hr_get(self, key):
#         if self.lr_size is None:
#             raise ValueError('please run lr_get first!!!')
#         hr_img = Image.open(self.hr_env[key])
#         hr_size = [x * self.scale_factor for x in self.lr_size]
#         hr_img = hr_img.resize(hr_size, Image.BICUBIC)
#         hr_img = self.to_tensor(hr_img)
#         return hr_img
