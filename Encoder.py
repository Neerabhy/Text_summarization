from utils import *
import torch
import os
from os.path import exists
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)
        self.feed_forward = feed_forward
        self.attn = self_attn
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x:self.attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


