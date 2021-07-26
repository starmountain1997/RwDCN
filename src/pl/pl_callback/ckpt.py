import pytorch_lightning as pl


class SSIMCkptCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, dirpath, save_top_k=5):
        super(SSIMCkptCallback, self).__init__(monitor='val_ssim', dirpath=dirpath,
                                               filename='{epoch:02d}-{val_ssim:.4f}', save_top_k=save_top_k, mode='max')


class LossCkptCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, dirpath, save_top_k=5):
        super(LossCkptCallback, self).__init__(monitor='val_loss', dirpath=dirpath,
                                               filename='{epoch:02d}-{val_loss:.4f}', save_top_k=save_top_k, mode='min')
