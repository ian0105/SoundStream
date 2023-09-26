from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from lightning.data import Maestro
import torch.nn as nn
import os


class LightningData(LightningDataModule):
    def __init__(
        self,
        path: str = None,
        batch_size: int = 16,
        segment_time: int = 3,
        sampling_rate: int = 24000,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        factory_kwargs = {
            "segment_time": self.hparams.segment_time,
            "sampling_rate": self.hparams.sampling_rate,
        }

        if stage == "fit":
            if self.hparams.path is not None:
                train_dataset = Maestro(
                    path=self.hparams.path, split="train", **factory_kwargs
                )
                self.train_dataset = train_dataset
            else:
                raise ValueError("path를 입력하시오")
            self.train_dataset = train_dataset

        if stage == "validate" or stage == "fit":
            if self.hparams.path is not None:
                valid_dataset = Maestro(
                    path=self.hparams.path, split="valid", **factory_kwargs
                )
                self.valid_dataset = valid_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
        )

    def collate_fn(self, batch):
        lengths = torch.tensor([elem.shape[-1] for elem in batch])
        return nn.utils.rnn.pad_sequence(batch, batch_first=True), lengths
