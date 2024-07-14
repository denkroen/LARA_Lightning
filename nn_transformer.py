import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class TransformerEncoderM(nn.module):
    def __init__(self, embedding_channels, window_length, num_layers, enc_hidden_neurons):
        super().__init__()

        self.n_head = self.get_nhead(embedding_channels, 8)

        self.cls_token = nn.Parameter(torch.zeros((1, embedding_channels)))
        self.position_embed = nn.Parameter(torch.randn(window_length + 1, 1, embedding_channels))
        
        #set transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model = embedding_channels, nhead = 8, dim_feedforward = enc_hidden_neurons,
                                       dropout = 0.1, activation = 'gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers = num_layers, norm = nn.LayerNorm(embedding_channels))
        
        # initialisation of parameters
        
        #for p in self.parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_uniform_(p)

        
    def forward(self, x):


        cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([cls_token, x])
        
        #position embedding
        x += self.position_embed
            
        # Transformer Encoder pass
        x = self.transformer_encoder(x)[0]

        return x

    def get_nhead(self, embed_dim, n_head):
        for hd in range(n_head, 0, -1):
            if embed_dim % hd == 0:
                return hd