import torch
import pytorch_lightning as pl

from HARWindows import HARWindows
from torch.utils.data import DataLoader



class LARADataModule(pl.LightningDataModule):

    def __init__(self, datadir, batch_size, num_workers):

        self.data_dir = datadir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        pass

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.test_ds = HARWindows(csv_file=self.data_dir+"test.csv", root_dir=self.data_dir+"sequences_test/")
        self.val_ds = HARWindows(csv_file=self.data_dir+"val.csv",root_dir=self.data_dir+"sequences_val/")
        self.train_ds  = HARWindows(csv_file=self.data_dir+"train.csv",root_dir=self.data_dir+"sequences_train/")



        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )