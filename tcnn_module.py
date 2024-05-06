import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision
from nn_tcnn_convolutions import TCNNConvBlock
from nn_tcnn_classificator import ClassificationHead
import numpy as np


class TCNN(pl.LightningModule):
    def __init__(self, learning_rate, num_filters, filter_size, mode, num_attributes, num_classes, window_length, sensor_channels, path_attributes):
        super().__init__()
        #TODO: move some functions to utils.py
        self.lr = learning_rate #def schedule

        latent_size = self.compute_feature_map_size(0,window_length,sensor_channels) 

        self.mode = mode

        if self.mode == "attribute":
            self.loss = nn.BCELoss()
            output_neurons = num_attributes

            # load attribute mapping
            self.attr = self.reader_att_rep(path_attributes) 
            for attr_idx in range(self.attr.shape[0]):
                self.attr[attr_idx, 1:] = self.attr[attr_idx, 1:] / np.linalg.norm(self.attr[attr_idx, 1:])

            self.atts = torch.from_numpy(self.attr).type(dtype=torch.FloatTensor)
            self.atts = self.atts.type(dtype=torch.cuda.FloatTensor)
                        
        elif self.mode == "classification":
            self.loss_shape = nn.CrossEntropyLoss()
            output_neurons = num_classes



        self.conv = TCNNConvBlock(1, num_filters, filter_size)
        self.classificator = ClassificationHead(latent_size, output_neurons, [128,128])

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        )



    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(x.size()[0], x.size()[1], x.size()[2])
        x = self.classificator.forward(x)

        if self.mode == "attribute":
            x = F.sigmoid(x)
        elif self.mode == "classification":
            x = F.softmax(x)

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
            pred = self.efficient_distance(pred) #distances
            pred = self.atts[torch.argmin(pred, dim=1), 0] #classes

        return loss, pred, label
    
    def efficient_distance(self, predictions):
        '''
        Compute Euclidean distance from predictions (output of sigmoid) to attribute representation

        @param predictions: torch array with predictions (output from sigmoid)
        @return distances: Euclidean Distance to each of the vectors in the attribute representation
        '''
        dist_funct = torch.nn.PairwiseDistance()

        # Normalize the predictions of the network
        for pred_idx in range(predictions.size()[0]):
            predictions[pred_idx, :] = predictions[pred_idx,:] / torch.norm(predictions[pred_idx, :])

        predictions = predictions.repeat(self.attr.shape[0], 1, 1)
        predictions = predictions.permute(1, 0, 2)

        # compute the distance among the predictions of the network
        # and the the attribute representation
        distances = dist_funct(predictions[0], self.atts[:, 1:])
        distances = distances.view(1, -1)
        for i in range(1, predictions.shape[0]):
            dist = dist_funct(predictions[i], self.atts[:, 1:])
            distances = torch.cat((distances, dist.view(1, -1)), dim=0)

        
        # return the distances
        return distances
    
    def compute_feature_map_size(self, padding, channels, window_size):
        # Computing the size of the feature maps
        Wx, Hx = self.size_feature_map(Wx=window_size,
                                       Hx=channels,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        return Wx, Hx

    
    def size_feature_map(self, Wx, Hx, F, P, S, type_layer = 'conv'):
        '''
        Computing size of feature map after convolution or pooling

        @param Wx: Width input
        @param Hx: Height input
        @param F: Filter size
        @param P: Padding
        @param S: Stride
        @param type_layer: conv or pool
        @return Wy: Width output
        @return Hy: Height output
        '''

        if self.config["fully_convolutional"] == "FCN":
            Pw = P[0]
            Ph = P[1]
        elif self.config["fully_convolutional"] == "FC":
            Pw = P
            Ph = P

        if type_layer == 'conv':
            Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

        elif type_layer == 'pool':
            Wy = 1 + (Wx - F[0]) / S[0]
            Hy = 1 + (Hx - F[1]) / S[1]

        return Wy, Hy
    
    def reader_att_rep(self, path: str) -> np.array:
        '''
        gets attribute representation from txt file.

        returns a numpy array

        @param path: path to file
        @param att_rep: Numpy matrix with the attribute representation
        '''

        att_rep = np.loadtxt(path, delimiter=',', skiprows=1)
        return att_rep


    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=self.lr)