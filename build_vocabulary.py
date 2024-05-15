from utils import *
import torch
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

def build_vocabulary(spacy_en, spacy_hi):
    def tokenize_en(text): #Tokenizer for text 
        return tokenize(text, spacy_en)
    dataset = pd.read_csv('path for your dataset')
    
    #dataset = dataset[~pd.isnull(dataset['article column name'])]
    #dataset_ar = data['article']
    #dataset_hi = data['highlights'] 
    print("Building article vocabulary.")
    vocab_src = build_vocab_from_iterator(
                yield_tokens(dataset_ar, tokenize_hi),
                min_freq=2,
                specials=["<s>", "</s>", "<blank>", "<unk>"],)
    print("Building highlight vocabulary.")
    vocab_tgt = build_vocab_from_iterator(
                yield_tokens(dataset_ar, tokenize_hi),
                min_freq=2,
                specials=["<s>", "</s>", "<blank>", "<unk>"],)
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])
    return vocab_src, vocab_tgt

def load_vocab(spacy_en, spacy_hi):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_en, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt
