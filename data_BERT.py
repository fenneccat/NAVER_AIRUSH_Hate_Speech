## Counter({0: 70181, 1: 29819})

import json
import typing
from typing import List, Dict, Tuple
from pydoc import locate
from collections import Counter
from collections import defaultdict

from torchtext.data import Dataset, Example, Field, LabelField
from torchtext.vocab import Vocab
import torch
from nsml.constants import DATASET_PATH

import random


class HateSpeech(object):
    MAX_LEN = 512
    UNK_TOKEN = 0  # '<unk>'
    PAD_TOKEN = 1  # '<pad>'
    SPACE_TOKEN = 2  # '<sp>'
    INIT_TOKEN = 3  # '<s>'
    EOS_TOKEN = 4  # '<e>'
    MASKED_TOKEN = 6015
    TOKENS = [PAD_TOKEN, UNK_TOKEN, SPACE_TOKEN, INIT_TOKEN, EOS_TOKEN]
    FIELDS_TOKEN_ATTRS = ['init_token', 'eos_token', 'unk_token', 'pad_token']
    FIELDS_ATTRS = FIELDS_TOKEN_ATTRS + ['sequential', 'use_vocab', 'fix_length']

    VOCAB_PATH = 'fields.json'

    def __init__(self, corpus_path=None, split: Tuple[int, int] = None, raw=False):
        if raw: self.fields, self.max_vocab_indexes = self.load_fields_raw(self.VOCAB_PATH)
        else: self.fields, self.max_vocab_indexes = self.load_fields(self.VOCAB_PATH)

        if corpus_path:
            if raw: self.examples = self.load_corpus_raw(corpus_path)
            else: self.examples = self.load_corpus(corpus_path)

            if split:
                total = len(self.examples)
                pivot = int(total / sum(split) * split[0])
                self.datasets = [Dataset(self.examples[:pivot], fields=self.fields),
                                 Dataset(self.examples[pivot:], fields=self.fields)]
            else:
                self.datasets = [Dataset(self.examples, fields=self.fields)]

    def load_corpus(self, path) -> List[Example]:
        preprocessed = []
        with open(path) as fp:
            for line in fp:
                if line:
                    ex = Example()
                    for k, v in json.loads(line).items():
                        setattr(ex, k, v)
                    preprocessed.append(ex)

        return preprocessed

    def load_corpus_raw(self, path) -> List[Example]:
        MASKED_TOKEN = 6015
        preprocessed = []
        with open(path) as fp:
            for line in fp:
                if line:
                    read_line = json.loads(line)
                    if len(read_line) > 1:
                        ex = Example()
                        for k, v in json.loads(line).items():
                            setattr(ex, k, v)
                    else:
                        ex = Example()
                        v = read_line['syllable_contents']
                        setattr(ex, 'original_sample', v)

                        length = len(v)
                        #masked_idx = random.choice(range(length))
                        portion = int(length*0.15)
                        masked_idxes = random.sample(range(length), portion)
                        for masked_idx in masked_idxes:
                            v[masked_idx] = MASKED_TOKEN

                        setattr(ex, 'masked_sample', v)
                        #setattr(ex, 'token_type_ids', [0]*length)
                        #setattr(ex, 'attention_mask', [1]*length)
                    preprocessed.append(ex)

        return preprocessed

    def dict_to_field(self, dicted: Dict) -> Field:
        field = locate(dicted['type'])(dtype=locate(dicted['dtype']))
        for k in self.FIELDS_ATTRS:
            setattr(field, k, dicted[k])

        if 'vocab' in dicted:
            v_dict = dicted['vocab']
            vocab = Vocab()
            vocab.itos = v_dict['itos']
            vocab.stoi.update(v_dict['stoi'])
            vocab.unk_index = v_dict['unk_index']
            if 'freqs' in v_dict:
                vocab.freqs = Counter(v_dict['freqs'])
        else:
            vocab = Vocab(Counter())
            field.use_vocab = False
        field.vocab = vocab

        return field

    def load_fields(self, path) -> Dict[str, Field]:
        loaded_dict = json.loads(open(path).read())
        for k, v in loaded_dict.items():
            loaded_dict[k]['max_vocab_index'] += 1
        max_vocab_indexes = {k: v['max_vocab_index'] for k, v in loaded_dict.items()}
        return {k: self.dict_to_field(v) for k, v in loaded_dict.items()}, max_vocab_indexes

    def load_fields_raw(self, path) -> Dict[str, Field]:
        #loaded_dict = json.loads(open(path).read())
        loaded_dict = dict()
        loaded_dict['original_sample'] = {'type': 'torchtext.data.field.Field', 'dtype': 'torch.int64',
                                          'init_token': None, 'eos_token': None, 'unk_token': 0, 'pad_token': 1,
                                          'sequential': True, 'use_vocab': True, 'fix_length': None,
                                          'batch_first': True, 'max_vocab_index': 6015}

        loaded_dict['masked_sample'] = {'type': 'torchtext.data.field.Field', 'dtype': 'torch.int64',
                                          'init_token': None, 'eos_token': None, 'unk_token': 0, 'pad_token': 1,
                                          'sequential': True, 'use_vocab': True, 'fix_length': None,
                                          'batch_first': True, 'max_vocab_index': 6015}

        max_vocab_indexes = {k: v['max_vocab_index'] for k, v in loaded_dict.items()}
        return {k: self.dict_to_field(v) for k, v in loaded_dict.items()}, max_vocab_indexes
