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
from transformers import BertForSequenceClassification, BertForMaskedLM
from transformers import BertConfig, PretrainedConfig
from transformers import AdamW


import torch.nn.functional as F

from model import BaseLine, BaseLine_two_class
from data_BERT import HateSpeech

def bind_model(model):
    def save(dirname, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))

    def load(dirname, *args):
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        model.load_state_dict(checkpoint['model'])

    def infer(raw_data, **kwargs):
        model.eval()
        examples = HateSpeech(raw_data, raw=True).examples
        tensors = [torch.tensor(ex.original_sample, device='cuda').reshape([-1, 1]) for ex in examples]
        results = [model(ex).tolist() for ex in tensors]
        return results

    nsml.bind(save=save, load=load, infer=infer)

class BERTMLMTrainer(object):
    #TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)
    TRAIN_DATA_PATH = '{}/train/raw.json'.format(DATASET_PATH)

    def __init__(self, hdfs_host: str = None, device: str = 'cpu'):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1), raw=True)

        self.configuration = BertConfig(vocab_size = self.task.max_vocab_indexes['original_sample'], num_hidden_layers = 2)

        self.model = BertForMaskedLM(self.configuration)
        self.model.to(self.device)
        self.loss_fn = nn.BCELoss()
        self.batch_size = 128
        self.__test_iter = None
        bind_model(self.model)

    @property
    def test_iter(self) -> Iterator:
        if self.__test_iter:
            self.__test_iter.init_epoch()
            return self.__test_iter
        else:
            self.__test_iter = Iterator(self.task.datasets[1], batch_size=self.batch_size, repeat=False,
                                        sort_key=lambda x: len(x.original_sample), train=False,
                                        device=self.device)
            return self.__test_iter

    def train(self):
        print("THIS IS BERT Masked LM TRAINER")

        max_epoch = 32
        #optimizer = optim.Adam(self.model.parameters())
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        total_len = len(self.task.datasets[0])
        ds_iter = Iterator(self.task.datasets[0], batch_size=self.batch_size, repeat=False,
                           sort_key=lambda x: len(x.original_sample), train=True, device=self.device)
        min_iters = 10000
        for epoch in range(max_epoch):
            loss_sum, prediction_score, len_batch_sum = 0., 0., 0.
            ds_iter.init_epoch()
            tr_total = math.ceil(total_len / self.batch_size)
            #tq_iter = tqdm(enumerate(ds_iter), total=tr_total, miniters=min_iters, unit_scale=self.batch_size,
            #              bar_format='{n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}] {desc}')

            self.model.train()
            for i, batch in enumerate(ds_iter):
                self.model.zero_grad()
                outputs = self.model(batch.masked_sample.transpose(0,1).contiguous(), labels = batch.original_sample.transpose(0,1).contiguous())

                loss = outputs[0]
                loss.backward()
                optimizer.step()

                len_batch = len(batch)
                len_batch_sum += len_batch
                #prediction_score += torch.sum(outputs[1], dtype=torch.float32).tolist()
                loss_sum += loss.tolist() * len_batch
                if i % min_iters == 0:
                    pass
                    #tq_iter.set_description('{:2} loss: {:.5}'.format(epoch, loss_sum / len_batch_sum), True)
                if i == 3000:
                    break

            #tq_iter.set_description('{:2} loss: {:.5}'.format(epoch, loss_sum / total_len), True)

            print(json.dumps(
                {'type': 'train', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_sum / total_len}))

            loss_avg, te_total = self.eval(self.test_iter, len(self.task.datasets[1]))

            print(json.dumps(
                {'type': 'test', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_avg}))
            nsml.save(epoch)
            self.save_model(self.model, 'e{}'.format(epoch))

    def eval(self, iter:Iterator, total:int) -> (List[float], float, List[float], int):
        #tq_iter = tqdm(enumerate(iter), total=math.ceil(total / self.batch_size),
        #              unit_scale=self.batch_size, bar_format='{r_bar}')
        pred_score_lst = list()
        loss_sum= 0.
        acc_lst = list()

        self.model.eval()
        for i, batch in enumerate(iter):
            outputs = self.model(batch.masked_sample.transpose(0,1).contiguous(), labels = batch.original_sample.transpose(0,1).contiguous())

            losses = outputs[0]

            loss_sum += losses.tolist() * len(batch)

        return loss_sum / total, total

    def save_model(self, model, appendix=None):
        file_name = 'model'
        if appendix:
            file_name += '_{}'.format(appendix)
        torch.save({'model': model, 'task': type(self.task).__name__}, file_name)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pause', default=0)
    args = parser.parse_args()
    if args.pause:
        task = HateSpeech(raw=True)
        configuration = BertConfig(vocab_size=task.max_vocab_indexes['original_sample'], num_hidden_layers = 2)

        model = BertForMaskedLM(configuration)
        model.to("cuda")
        bind_model(model)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        trainer = BERTMLMTrainer(device='cuda')
        trainer.train()