import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy


def cli_main():
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            'accelerator': 'gpu',
            'strategy': 'ddp_find_unused_parameters_true',
            'log_every_n_steps': 100,
            'callbacks': [
                ModelCheckpoint(
                    monitor='val_loss_g',
                    mode='min',
                    save_top_k=3,
                    save_last=True,
                    every_n_epochs=1,
                    filename='{epoch}-{step}-{val_loss}',
                ),
                LearningRateMonitor(logging_interval='step'),
                ModelSummary(max_depth=4)
            ]
        }
    )


if __name__ == "__main__":
    cli_main()