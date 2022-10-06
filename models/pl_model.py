import pytorch_lightning as pl
import torch
from conformer import ConformerModel
from data.data_module import SpeechCommandsDataModule

class ConformerModelPyTorchLightning(pl.LightningModule):
    def __init__(self, dim=128, lr=1e-4):
        super().__init__()
        self.model = ConformerModel(dim=dim)
        # there are 35 classes in the dataset
        # the output of the model is the same as the input

        self.classifier = None # TODO: Add classifier layer
        self.lr = lr

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

def main():
    model = ConformerModelPyTorchLightning()
    data = SpeechCommandsDataModule(batch_size=32, num_workers=4)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=100)
    trainer.fit(model, data)

if __name__ == '__main__':
    main()