import torch.nn.functional as F
from torch import nn, optim

#TODO


class AutoencoderClassificationHead(nn.module):
    def __init__(self, latent_size, num_classes, neuron_config=[128,128]):
        super().__init__()

        self.fc1 = nn.Linear(latent_size, neuron_config[0])
        self.fc2 = nn.Linear(neuron_config[0], neuron_config[1])
        self.fc3 = nn.Linear(neuron_config[1], num_classes)

        self.pooling = nn.AvgPool2d(kernel_size=[2,1], stride=[1,1])


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
