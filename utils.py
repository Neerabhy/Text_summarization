import torch
import os
from os.path import exists
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import spacy
import pandas as pd

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_length = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x- mean) / (std + self.eps) + self.b_2

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal = 1).type(torch.uint8)
    return subsequent_mask ==0

class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


spacy_en = spacy.load("en_core_web_sm") 
def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]
    
def yield_tokens(data_iter, tokenizer):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple)
