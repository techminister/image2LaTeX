import pandas as pd
import torch
from models import device


class Vocabulary:
    def __init__(self, path="./data_props.pkl", delimiter=" "):
        props = pd.read_pickle(path)
        self.delimiter = delimiter if delimiter else ""

        self.word2index = props.get("word2id")
        self.index2word = props.get("id2word")
        self.EOS = props.get("NullTokenID")
        self.SOS = props.get("StartTokenID")
        self.n_words = props.get("K")

    def tokenize(self, sentence):
        if self.delimiter:
            return sentence.split(self.delimiter)
        else:
            return list(sentence)

    def detokenize(self, sentence):
        return self.delimiter.join(sentence)

    def encode(self, sentence, text=False):
        indices = (
            [self.word2index[word] for word in self.tokenize(sentence)]
            if text
            else sentence
        )
        return torch.tensor(indices + [self.EOS], dtype=torch.long, device=device)

    def decode(self, indices):
        sentence = [self.index2word[ind] for ind in indices]
        return self.detokenize(sentence)

    def __len__(self):
        return self.n_words
