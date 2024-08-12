import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


def load_data():
    train_iter, test_iter = IMDB(root='./data')
 
    tokenizer = get_tokenizer('basic_english')
    
   
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    return train_iter, test_iter, tokenizer, vocab
