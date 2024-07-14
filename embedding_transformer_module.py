import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from nn_transformer import TransformerEncoderM
import numpy as np

from utils import efficient_distance, select_embedding_generator

class EmbeddingTransformer(pl.LightningModule):
    def __init__(self, learning_rate, num_filters, filter_size, mode, num_attributes, num_classes, window_length, sensor_channels, path_attributes, generator, n_trans_layers, n_trans_hidden_neurons, embedding_size):
        super().__init__()


        #TODO: rework training methods, add embedding rotation in generator modules
        self.lr = learning_rate #def schedule

        #embedding generator
        self.embedding_generator = select_embedding_generator(generator, window_length,sensor_channels, filter_size, num_classes, num_filters, embedding_size)
        embedding_channels = self.embedding_generator.get_embedding_size()

        #transformer
        self.transformer = TransformerEncoderM(embedding_channels, window_length, n_trans_layers, n_trans_hidden_neurons)

        #classificator
        self.activation_function = nn.GELU()
        self.imu_head = nn.Sequential(nn.LayerNorm(embedding_channels), nn.Linear(embedding_channels, embedding_channels//4),
                                      self.activation_function, nn.Dropout(0.1), nn.Linear(embedding_channels//4, num_classes))
        self.softmax = nn.Softmax()



        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        )



    def forward(self, x):

        x = self.embedding_generator.forward(x)
        x = self.transformer.forward(x)
        x = self.imu_head.forward(x)




        return x

    def training_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        train_acc = self.accuracy(prediction, label)
        train_f1 = self.f1(prediction, label)

        self.log_dict(
            {
                "train_loss": loss,
                "train_prediction": prediction,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        

        return loss

    def validation_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        #calc_metrics
        val_acc = self.accuracy(prediction, label)
        val_f1 = self.f1(prediction, label)


        self.log_dict(
            {
                "validation_loss": loss,
                "validation_prediction": prediction,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        return loss

    def test_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        test_acc = self.accuracy(prediction, label)
        test_f1 = self.f1(prediction, label)

        self.log_dict(
            {
                "validation_loss": loss,
                "validation_accuracy_shape": prediction,

            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        
        input_tensor = batch["data"]
        label = batch["label"]
        label = label.reshape(-1)

        pred = self.forward(input_tensor)
        loss = self.loss(pred, label) 



        if self.mode == "attribute":
            pred = efficient_distance(self.attr, self.atts, pred) #distances
            pred = self.atts[torch.argmin(pred, dim=1), 0] #classes

        return loss, pred, label
    
    

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=self.lr)