import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
import torchvision

from ..utils import create_optimizer_v2


def get_model(model_name: str, num_classes: int):
    all_model = timm.list_models()
    if model_name not in all_model:
        raise ValueError(f'model name must in {all_model}')

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    return model


class ClassificationModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            classes_num: int,
            logger
    ):
        super(ClassificationModel, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.classes_num = classes_num
        self.wandb_logger = logger
        self.model = self.load_model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = torchmetrics.Accuracy()
        self.result = []

    def load_model(self):
        return get_model(
            model_name=self.model_name,
            num_classes=self.classes_num
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        img, label = batch
        out = self.forward(img)
        loss = self.loss_fn(out, label)
        self.log('train_loss', value=loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)
        loss = self.loss_fn(out, y)

        sample_imgs = x[:6]
        grid = torchvision.utils.make_grid(sample_imgs)

        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        self.log('val_loss', value=loss, on_epoch=True)
        self.log('val_acc', value=val_acc, on_epoch=True)
        self.wandb_logger.log_image(key=''.join(y[:6]), images=[grid])

    def configure_optimizers(self):
        return create_optimizer_v2(self.parameters(), opt='sgd', lr=0.01)
