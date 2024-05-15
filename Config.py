from utils import *
from DataLoader import *
from Model import *
import torch
import os
from os.path import exists
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import pandas as pd
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
config = {
        "batch_size": 64,
        "distributed": False,
        "num_epochs": 10,
        "accum_iter": 10,
        "base_lr": 0.005,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
d_model = 512
device = torch.device("cuda" if torch.cuda.is_available else 'cpu')
model = make_model(len(vocab_src), len(vocab_tgt), N = 6)
optimizer = torch.optim.Adam(model.parameters(), lr = config["base_lr"],
                                betas = (0.9, 0.98), eps = 1e-9)
lr_scheduler = LambdaLR(optimizer = optimizer,
                        lr_lambda = lambda step :rate(step, d_model, factor = 1, warmup = config["warmup"]),)

model.to(device)
train_model(model, vocab_src, vocab_tgt, spacy_en, spacy_hi, config, optimizer, lr_scheduler)
