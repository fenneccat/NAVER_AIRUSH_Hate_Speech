## Counter({0: 70181, 1: 29819})

import sys
import math
import os
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join

from tqdm import tqdm
import numpy as np
import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

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

from os import listdir
from os.path import isfile, join


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

    def __init__(self, corpus_path=None):
        self.load_corpus(corpus_path)

    def load_corpus(self, path):
        preprocessed = []

        with open(path) as fp:
            for line in fp:
                if line:
                    line_data = json.loads(line)
                    print(line_data)
                    print(type(line_data))
                break

if __name__ == '__main__':
    print("hihi")
    #TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)
    #print(TRAIN_DATA_PATH)
    path = os.path.join('./', 'train_unlabeled')
    os.mkdir(path)
    #empty_list = []
    #with open("./train_unlabeled/empty.json", "w") as json_file:
    #    json.dump(empty_list, json_file)
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    print(onlyfiles)
    #HateSpeech(TRAIN_DATA_PATH)