import torch.nn.functional as F
from torch import nn, optim


class MLP(nn.module):
    def __init__(self, input_size, embedding_size, neuron_config=[128,128]):
        super().__init__()
        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(input_size, neuron_config[0])
        self.fc2 = nn.Linear(neuron_config[0], neuron_config[1])
        self.fc3 = nn.Linear(neuron_config[1], embedding_size)

    def forward(self, x):
        pred = 0
        reconst = 0

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embedding = F.relu(self.fc3(x))

        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size
