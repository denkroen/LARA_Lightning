import torch.nn.functional as F
from torch import nn, optim

class TCNNConvBlock(nn.module):
    def __init__(self, in_channels, num_filters, filter_size):
        super().__init__()

        padding = 0

        self.conv1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=num_filters,
                                     kernel_size=(filter_size, 1),
                                     stride=1, padding=padding)
        self.conv1_2 = nn.Conv2d(in_channels=num_filters,
                                     out_channels=num_filters,
                                     kernel_size=(filter_size, 1),
                                     stride=1, padding=padding)
        self.conv2_1 = nn.Conv2d(in_channels=num_filters,
                                     out_channels=num_filters,
                                     kernel_size=(filter_size, 1),
                                     stride=1, padding=padding)
        self.conv2_2 = nn.Conv2d(in_channels=num_filters,
                                     out_channels=num_filters,
                                     kernel_size=(filter_size, 1),
                                     stride=1, padding=padding)

    def forward(self, x):

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        return x

