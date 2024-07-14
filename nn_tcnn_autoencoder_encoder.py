import torch.nn.functional as F
from torch import nn, optim
from nn_tcnn_convolutions import TCNNConvBlock

class TCNNEncoder(nn.module):
    def __init__(self, in_channels, num_filters, filter_size):
        super().__init__()

        padding = 0

        self.tcnn_conv = TCNNConvBlock(in_channels, num_filters, filter_size)

        self.conv1_1 = nn.Conv2d(in_channels=num_filters,
                                        out_channels=256,
                                        kernel_size=(1,1),
                                        stride=1, padding=padding)
        self.conv1_2 = nn.Conv2d(in_channels=256,
                                        out_channels=256,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=padding)

        
    def forward(self, x):

        x = self.tcnn_conv(x)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        return x

