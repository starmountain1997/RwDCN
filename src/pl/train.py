import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.reds import REDSDataModule
from src.pl import model_dict, config_path
from src.pl.callback import SaveResultCallback, SSIMCkptCallback


@hydra.main(config_path=config_path, config_name='train.yaml')
def train(cfg: DictConfig) -> None:
    pl.seed_everything(8823)# Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random. 8823 is meaningless. 
    save_test_result_callback = SaveResultCallback(img_save_dir=cfg.log.img_save_dir)
    ckpt_save_callback = SSIMCkptCallback(dirpath=cfg.log.net_save_dir)
    tb_logger = TensorBoardLogger(save_dir=cfg.log.tb_save_dir, name=cfg.run_name)
    trainer = pl.Trainer(
        gpus=cfg.experiment.gpus,
        accelerator='ddp',
       accumulate_grad_batches=cfg.experiment.accumulate_grad_batches,
        val_check_interval=1024,
        resume_from_checkpoint=cfg.model.ckpt,
        callbacks=[save_test_result_callback, ckpt_save_callback],
        default_root_dir=cfg.log.log_dir,
        logger=tb_logger
    )
    model = model_dict[cfg.model.name]()
    reds = REDSDataModule(dataset_type=cfg.model.dataset_type, lr_size=cfg.model.lr_size,
                          batch_size=cfg.model.batch_size)
    trainer.fit(model, reds)


if __name__ == '__main__':
    train()
