import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.cufed5 import CUFED5DataModule
from src.data.reds import REDSDataModule
from src.data.vid4 import Vid4DataModule
from src.pl import config_path, model_dict
from src.pl.callback import SaveResultCallback


@hydra.main(config_path=config_path, config_name='test.yaml')
def test(cfg: DictConfig) -> None:
    pl.seed_everything(8823)
    save_test_result_callback = SaveResultCallback(img_save_dir=cfg.log.img_save_dir)
    tb_logger = TensorBoardLogger(save_dir=cfg.log.tb_save_dir, name=cfg.run_name)
    trainer = pl.Trainer(
        gpus=[0],
        callbacks=[save_test_result_callback],
        logger=tb_logger
    )
    model = model_dict[cfg.model.name]()
    model = model.load_from_checkpoint(cfg.model.ckpt, strict=False)
    if cfg.model.dataset_name == 'vid4':
        dm = Vid4DataModule(dataset_type=cfg.model.dataset_type)
    elif cfg.model.dataset_name == 'reds':
        dm = REDSDataModule(dataset_type=cfg.model.dataset_type, batch_size=1, lr_size=cfg.model.lr_size)
    elif cfg.model.dataset_name == 'cufed5':
        dm = CUFED5DataModule(dataset_type=cfg.model.dataset_type)
    else:
        raise ValueError(f'invalid dataset_name: {cfg.model.dataset_name}')
    trainer.test(model, dm.test_dataloader())


if __name__ == '__main__':
    test()
