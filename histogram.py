import sys
import json
import math
import typing
from typing import Dict, List
import os
from argparse import ArgumentParser

from torch import nn, optim
import torch
from torchtext.data import Iterator
from tqdm import tqdm
import numpy as np
import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from torchtext.data import Example
from matplotlib import pyplot as plt


def bind_json(examples):
    def save(dirname, *args):
        with open(os.path.join(dirname, 'raw_labeled.json'), "w") as json_file:
            json.dump(examples, json_file)

    def load(dirname, *args):
        with open(os.path.join(dirname, 'raw_labeled.json'), "r") as json_file:
            examples = json.load(json_file)

    def infer(raw_data, **kwargs):
        pass

    nsml.bind(save=save, load=load, infer=infer)

preds = []
bind_json(preds)
nsml.load(checkpoint='raw_data', session='fenneccat/hate_3/458')


def count(list1, l, r):
    c = 0
    # traverse in the list1
    for x in list1:
        # condition check
        if x >= l and x < r:
            c += 1
    return c


# driver code
print(preds)
'''
for xx in range(0, 100, 5):
    x = xx/100
    print("{}~{}: {}".format(x, x+0.05,count(preds, x, x+0.05)))
'''
#plt.hist(preds)

#plt.show()