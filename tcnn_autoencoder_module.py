import torch.nn.functional as F
from torch import nn
from nn_tcnn_autoencoder_encoder import TCNNEncoder
from nn_tcnn_autoencoder_decoder import TCNNDecoder
from nn_tcnn_autoencoder_classificator import AutoencoderClassificationHead

from utils import compute_feature_map_size_tcnn



class TCNN(nn.module):
    def __init__(self, in_channels, window_length, sensor_channels, filter_size, num_classes, num_filters):
        super().__init__()
        W,H = compute_feature_map_size_tcnn(0,window_length,sensor_channels,filter_size)
        self.embedding_size = W
        latent_size = W*H*256

        self.encoder = TCNNEncoder(in_channels, num_filters, filter_size)
        self.decoder = TCNNDecoder(in_channels, num_filters, filter_size)
        self.classificator = AutoencoderClassificationHead(latent_size,num_classes)

        self.embedding_convolution = nn.Conv2d(in_channels=num_filters,
                                        out_channels=1,
                                        kernel_size=(1,1),
                                        stride=1, padding=0)


    def forward(self, x):
        x = self.encoder.forward(x)
        embedding = self.embedding_convolution(x)
        reconst = self.decoder.forward(x)

        x = x.view(x.size()[0], x.size()[1], x.size()[2])
        pred = self.classificator.forward(x)

        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size