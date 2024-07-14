import torch
import numpy as np
from nn_tcnn import TCNN
from nn_mlp import MLP

def select_embedding_generator(generator, window_length, sensor_channels, filter_size, num_classes, num_filters, embedding_size):
     if generator == "cnn":
          return TCNN(1, window_length, sensor_channels, filter_size, num_classes, num_filters)
     if generator == "mlp":
          input_size = window_length*sensor_channels
          return MLP(input_size, embedding_size)
     pass


def efficient_distance(attr, atts, predictions):
        '''
        Compute Euclidean distance from predictions (output of sigmoid) to attribute representation

        @param predictions: torch array with predictions (output from sigmoid)
        @return distances: Euclidean Distance to each of the vectors in the attribute representation
        '''
        dist_funct = torch.nn.PairwiseDistance()

        # Normalize the predictions of the network
        for pred_idx in range(predictions.size()[0]):
            predictions[pred_idx, :] = predictions[pred_idx,:] / torch.norm(predictions[pred_idx, :])

        predictions = predictions.repeat(attr.shape[0], 1, 1)
        predictions = predictions.permute(1, 0, 2)

        # compute the distance among the predictions of the network
        # and the the attribute representation
        distances = dist_funct(predictions[0], atts[:, 1:])
        distances = distances.view(1, -1)
        for i in range(1, predictions.shape[0]):
            dist = dist_funct(predictions[i], atts[:, 1:])
            distances = torch.cat((distances, dist.view(1, -1)), dim=0)

        
        # return the distances
        return distances
    
def compute_feature_map_size_tcnn(padding, channels, window_size, filter_size):
    # Computing the size of the feature maps
    Wx, Hx = size_feature_map(Wx=window_size,
                                    Hx=channels,
                                    F=(filter_size, 1),
                                    P=padding, S=(1, 1), type_layer='conv')
    Wx, Hx = size_feature_map(Wx=Wx,
                                    Hx=Hx,
                                    F=(filter_size, 1),
                                    P=padding, S=(1, 1), type_layer='conv')
    Wx, Hx = size_feature_map(Wx=Wx,
                                    Hx=Hx,
                                    F=(filter_size, 1),
                                    P=padding, S=(1, 1), type_layer='conv')
    Wx, Hx = size_feature_map(Wx=Wx,
                                    Hx=Hx,
                                    F=(filter_size, 1),
                                    P=padding, S=(1, 1), type_layer='conv')
    return Wx, Hx


def size_feature_map( Wx, Hx, F, P, S, type_layer = 'conv'):
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

 
    Pw = P
    Ph = P

    if type_layer == 'conv':
        Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
        Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

    elif type_layer == 'pool':
        Wy = 1 + (Wx - F[0]) / S[0]
        Hy = 1 + (Hx - F[1]) / S[1]

    return Wy, Hy

def reader_att_rep( path: str) -> np.array:
    '''
    gets attribute representation from txt file.

    returns a numpy array

    @param path: path to file
    @param att_rep: Numpy matrix with the attribute representation
    '''

    att_rep = np.loadtxt(path, delimiter=',', skiprows=1)
    return att_rep
