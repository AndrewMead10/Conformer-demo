import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import transforms

from SPEECHCOMMANDS import SPEECHCOMMANDS


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 num_workers,
                 time_mask_param=20,
                 freq_mask_param=20,
                 num_time_masks=2,
                 num_freq_masks=4,
                 noise_max_ratio=0.2,
                 time_shift_amount=1,
                 num_classes=35):
        super(SpeechCommandsDataModule, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = nn.Sequential(*[transforms.TimeMasking(
            time_mask_param=time_mask_param) for _ in range(num_time_masks)],
            *[transforms.FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(num_freq_masks)])

        # self.train = SPEECHCOMMANDS(
        #     root='./', subset='training', download=True, transform=self.transforms, num_classes=num_classes, noise_max_ratio=noise_max_ratio, time_shift_amount=time_shift_amount)
        self.test = SPEECHCOMMANDS(
            root='./', subset='testing', download=True, transform=self.transforms, num_classes=num_classes)
        self.valid = SPEECHCOMMANDS(
            root='./', subset='validation', download=True, transform=self.transforms, num_classes=num_classes)

    def train_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)