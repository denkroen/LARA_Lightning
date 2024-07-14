import torch.nn.functional as F
from torch import nn, optim

class TCNNDecoder(nn.module):
    def __init__(self, in_channels, num_filters, filter_size):
        super().__init__()

        padding = 0


        self.deconv_1 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=256,
                                            kernel_size=(1, 1),
                                            stride=1, padding=padding)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=256, kernel_size=(1, 1),
                                            stride=1, padding=padding)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=num_filters,
                                            kernel_size=(1, 1),
                                            stride=1, padding=padding)
        self.deconv_4_1 = nn.ConvTranspose2d(in_channels=num_filters,
                                            out_channels=num_filters,
                                            kernel_size=(filter_size, 1),
                                            stride=1, padding=padding)
        self.deconv_4_2 = nn.ConvTranspose2d(in_channels=num_filters,
                                            out_channels=num_filters,
                                            kernel_size=(filter_size, 1),
                                            stride=1, padding=padding)
        self.deconv_5_1 = nn.ConvTranspose2d(in_channels=num_filters,
                                            out_channels=num_filters,
                                            kernel_size=(filter_size, 1),
                                            stride=1, padding=padding)
        self.deconv_5_2 = nn.ConvTranspose2d(in_channels=num_filters,
                                            out_channels=in_channels,
                                            kernel_size=(filter_size, 1),
                                            stride=1, padding=padding)

    def forward(self, x):

        x = F.relu(self.deconv_1(x))
        x = F.relu(self.deconv_2(x))
        x = F.relu(self.deconv_3(x))
        x = F.relu(self.deconv_4_1(x))
        x = F.relu(self.deconv_4_2(x))
        x = F.relu(self.deconv_5_1(x))
        x = F.sigmoid(self.deconv_5_2(x))


        return x

