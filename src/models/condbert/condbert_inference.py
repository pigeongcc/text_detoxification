import os
import sys
import torch
import numpy as np
import pickle

from importlib import reload
from .condBERT import condbert
reload(condbert)

from tqdm.auto import tqdm, trange
from transformers import BertTokenizer, BertForMaskedLM
from .condBERT.condbert import CondBertRewriter
from importlib import reload


class CondBERT():

    def __init__(self, dir_path: str="") -> None:
        self.editor = self.load_model(dir_path)

    def load_model(self, dir_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda:0')

        # load the model
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)

        model = BertForMaskedLM.from_pretrained(model_name)

        model.to(device)

        # load vocabularies for spans detection
        vocab_root = dir_path + 'condBERT/vocab/'

        with open(vocab_root + "negative-words.txt", "r") as f:
            s = f.readlines()
        negative_words = list(map(lambda x: x[:-1], s))
        with open(vocab_root + "toxic_words.txt", "r") as f:
            ss = f.readlines()
        negative_words += list(map(lambda x: x[:-1], ss))

        with open(vocab_root + "positive-words.txt", "r") as f:
            s = f.readlines()
        positive_words = list(map(lambda x: x[:-1], s))

        with open(vocab_root + 'word2coef.pkl', 'rb') as f:
            word2coef = pickle.load(f)


        token_toxicities = []
        with open(vocab_root + 'token_toxicities.txt', 'r') as f:
            for line in f.readlines():
                token_toxicities.append(float(line))
        token_toxicities = np.array(token_toxicities)
        token_toxicities = np.maximum(0, np.log(1/(1/token_toxicities-1)))   # log odds ratio

        # discourage meaningless tokens
        for tok in ['.', ',', '-']:
            token_toxicities[tokenizer.encode(tok)][1] = 3

        for tok in ['you']:
            token_toxicities[tokenizer.encode(tok)][1] = 0

        # applying the model
        reload(condbert)

        editor = CondBertRewriter(
            model=model,
            tokenizer=tokenizer,
            device=device,
            neg_words=negative_words,
            pos_words=positive_words,
            word2coef=word2coef,
            token_toxicities=token_toxicities,
        )

        return editor

    def detoxify(self, sentence: str):
        return self.editor.translate(sentence, prnt=False)