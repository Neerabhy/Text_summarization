from utils import *
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

def collate_batch(batch, src_pipe, tgt_pipe, src_vocab, tgt_vocab, device, max_padding = 128, pad_id = 2):
    bs_id = torch.tensor([0], device = device)
    eos_id = torch.tensor([1], device = device)
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
    
        processed_src = torch.cat(
        [bs_id,
        torch.tensor(src_vocab(src_pipe(_src)),
                    dtype = torch.int64,
                    device = device,),
        eos_id,], 0,)
        processed_tgt = torch.cat(
        [bs_id,
        torch.tensor(tgt_vocab(tgt_pipe(_tgt)),
                    dtype = torch.int64,
                    device = device,),
        eos_id,], 0,)
        src_list.append(pad(processed_src,
                           (0, max_padding-len(processed_src),),
                           value = pad_id),)
        tgt_list.append(pad(processed_tgt,
                           (0, max_padding-len(processed_tgt)),
                           value = pad_id,))
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    
    return (src, tgt)


def create_dataloaders(device, vocav_src, vocab_tgt, spacy_en, spacy_hi,batch_size,max_padding = 128, is_distributed = True,):
    def tokenize_en(text):
        return tokenize(text, spacy_en)
    def tokenize_hi(text):
        return tokenize(text, spacy_hi)
    
    def collate_fn(batch):
        return collate_batch(batch,tokenize_en, tokenize_hi, vocab_src, vocab_tgt, device, max_padding = max_padding, pad_id = vocab_src.get_stoi()["<blank>"],)
    
    train_dataset = pd.read_csv('path for dataset')
    train_dataset = train_dataset[~pd.isnull(train_dataset['column name of article'])]
    train_dataset = train_dataset[["column name of article", "column name of highlights"]]
    file = []
    for  index, row in data.iterrows():
          file.append((row['article'],row['highlights']))
    print("len_file :") 
    print(len(file))
    file = to_map_style_dataset(file)
    
    train_sampler = (DistributedSampler(file) if is_distributed else None)
    
    train_dataloader = DataLoader(file,
                                 batch_size = batch_size,
                                 shuffle = (train_sampler is None),
                                 sampler = train_sampler,
                                 collate_fn = collate_fn,)
    print(len(train_dataloader))
     
    return train_dataloader
