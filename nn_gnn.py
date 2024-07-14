import torch.nn.functional as F
from torch import nn, optim


class GNN(nn.module):
    def __init__(self, input_size, embedding_size):
        super().__init__()

        self.get_embedding_size = 0
        

    def forward(self, x):
        pred = 0
        reconst = 0
        embedding = 0


        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size
