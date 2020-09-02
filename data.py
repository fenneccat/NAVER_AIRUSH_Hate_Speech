## Counter({0: 70181, 1: 29819})
#

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
    TOKENS = [PAD_TOKEN, UNK_TOKEN, SPACE_TOKEN, INIT_TOKEN, EOS_TOKEN]
    FIELDS_TOKEN_ATTRS = ['init_token', 'eos_token', 'unk_token', 'pad_token']
    FIELDS_ATTRS = FIELDS_TOKEN_ATTRS + ['sequential', 'use_vocab', 'fix_length']

    VOCAB_PATH = 'fields.json'

    def __init__(self, corpus_path=None, split: Tuple[int, int] = None, shuffle = False,
                 raw = False, copy = False, semi=False, examples=None, l_thr=0.5, r_thr=0.5):
        if raw:
            self.VOCAB_PATH = 'fields_raw.json'
        else:
            self.VOCAB_PATH = 'fields.json'
        self.fields, self.max_vocab_indexes = self.load_fields(self.VOCAB_PATH)

        if corpus_path and copy:
            path = corpus_path
            examples = []
            with open(path) as fp:
                for line in fp:
                    if line:
                        line_data = json.loads(line)
                        examples.append(line_data)
            self.examples = examples

        if not copy:
            if semi:
                train_preprocessed, test_preprocessed = self.load_corpus_semi(corpus_path, shuffle, examples, l_thr, r_thr)
                self.examples = train_preprocessed + test_preprocessed
                self.datasets = [Dataset(train_preprocessed, fields=self.fields),
                                 Dataset(test_preprocessed, fields=self.fields)]

            elif corpus_path:
                self.examples = self.load_corpus(corpus_path, shuffle)
                if split:
                    total = len(self.examples)
                    pivot = int(total / sum(split) * split[0])
                    self.datasets = [Dataset(self.examples[:pivot], fields=self.fields),
                                     Dataset(self.examples[pivot:], fields=self.fields)]
                else:
                    self.datasets = [Dataset(self.examples, fields=self.fields)]

    def load_corpus_semi(self, path, shuffle, examples=None, l_thr=0.5, r_thr=0.5) -> (List[Example], List[Example]):
        print('semi data process')
        print('lthr, rthr', l_thr, r_thr)
        preprocessed_labeled = []
        preprocessed_unlabeled_low = []
        preprocessed_unlabeled_high = []
        zeros = 0
        ones = 0
        maxpredscore = 0
        minpredscore = float('inf')
        for line in examples:
            #print(line)
            if 'prediction_score' not in line:
                ex = Example()
                if line['eval_reply'] == 0:
                    zeros += 1
                else:
                    ones += 1
                for k, v in line.items():
                    setattr(ex, k, v)
                preprocessed_labeled.append(ex)
            elif line['prediction_score'] < l_thr:
                #print("I'm less than lthr: ", line['prediction_score'])
                ex = Example()
                if line['eval_reply'] == 0: zeros += 1
                else: ones += 1
                for k, v in line.items():
                    if k == 'prediction_score': continue
                    setattr(ex, k, v)
                preprocessed_unlabeled_low.append(ex)
            elif line['prediction_score'] > r_thr:
                #print("I'm larger than rthr: ", line['prediction_score'])
                ex = Example()
                if line['eval_reply'] == 0:
                    zeros += 1
                else:
                    ones += 1
                for k, v in line.items():
                    if k == 'prediction_score': continue
                    setattr(ex, k, v)
                preprocessed_unlabeled_high.append(ex)

        print("all unlabeled sample label 0: {}".format(len(preprocessed_unlabeled_low)))
        print("all unlabeled sample label 1: {}".format(len(preprocessed_unlabeled_high)))
        #preprocessed_unlabeled_low = random.sample(preprocessed_unlabeled_low, min(100000,len(preprocessed_unlabeled_low)))
        #preprocessed_unlabeled_high = random.sample(preprocessed_unlabeled_high, min(100000,len(preprocessed_unlabeled_high)))
        print("selected unlabeled sample label 0: {}".format(len(preprocessed_unlabeled_low)))
        print("selected unlabeled sample label 1: {}".format(len(preprocessed_unlabeled_high)))
        preprocessed_unlabeled = preprocessed_unlabeled_low+preprocessed_unlabeled_high
        print("total label 0 data: {} ".format(zeros))
        print("total label 1 data {} ".format(ones))
        print("mininum pred score: {}".format(minpredscore))
        print("maximum pred score: {}".format(maxpredscore))

        total = len(preprocessed_unlabeled)
        pivot = int(total / 10 * 9)

        if shuffle:
            random.shuffle(preprocessed_labeled)
            random.shuffle(preprocessed_unlabeled)

        #train_preprocessed = preprocessed_unlabeled+preprocessed_labeled[:pivot]
        #test_preprocessed = preprocessed_labeled[pivot:]

        # train_preprocessed = preprocessed_unlabeled[:pivot]
        # test_preprocessed = preprocessed_unlabeled[pivot:]

        train_preprocessed = preprocessed_unlabeled
        test_preprocessed = preprocessed_labeled

        return train_preprocessed, test_preprocessed

    def load_corpus(self, path, shuffle) -> List[Example]:

        preprocessed = []
        with open(path) as fp:
            for line in fp:
                if line:
                    ex = Example()
                    for k, v in json.loads(line).items():
                        setattr(ex, k, v)
                    preprocessed.append(ex)

        if shuffle: random.shuffle(preprocessed)

        return preprocessed

    def load_corpus_2(self, path) -> List[Example]:
        preprocessed = defaultdict(list)
        label = 0
        with open(path) as fp:
            for line in fp:
                if line:
                    ex = Example()
                    for k, v in json.loads(line).items():
                        setattr(ex, k, v)
                        if k == 'eval_reply':
                            label = v
                    preprocessed[label].append(ex)

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
        #print(loaded_dict)
        max_vocab_indexes = {k: v['max_vocab_index'] for k, v in loaded_dict.items()}
        return {k: self.dict_to_field(v) for k, v in loaded_dict.items()}, max_vocab_indexes
