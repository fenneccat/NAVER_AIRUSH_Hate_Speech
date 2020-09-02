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

import torch.nn.functional as F

from model import BaseLine, BaseLine_two_class
from data import HateSpeech

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
        examples = HateSpeech(raw_data).examples
        tensors = [torch.tensor(ex.syllable_contents, device='cuda').reshape([-1, 1]) for ex in examples]
        results = [model(ex).tolist() for ex in tensors]

        return results

    nsml.bind(save=save, load=load, infer=infer)

def bind_json(examples):
    def save(dirname, *args):
        with open(os.path.join(dirname, 'raw_labeled.json'), "w") as json_file:
            json.dump(examples, json_file)

    def load(dirname, *args):
        with open(os.path.join(dirname, 'raw_labeled.json'), "r") as json_file:
            print("successfully load pseudo labeled data")
            examples.extend(json.load(json_file))
        print("data size:", len(examples))

    def infer(raw_data, **kwargs):
        pass

    nsml.bind(save=save, load=load, infer=infer)

def f1_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        '''Calculate F1 score. Can work with gpu tensors

        The original implmentation is written by Michal Haltuf on Kaggle.

        Returns
        -------
        torch.Tensor
            `ndim` == 1. 0 <= val <= 1

        Reference
        ---------
        - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

        '''

        y_pred = torch.round(y_pred)
        #y_pred = (y_pred>0.6).to(torch.int64)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        #f1 = f1.clamp(min=epsilon, max=1 - epsilon)
        #f1.requires_grad = is_training
        return f1.mean()

class Trainer(object):
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)

    def __init__(self, hdfs_host: str = None, device: str = 'cpu', semi=False):
        self.device = device
        self.semi = semi
        if semi:
            examples = []
            bind_json(examples)
            nsml.load(checkpoint='raw_data', session='fenneccat/hate_3/621')

            self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1), semi=semi, examples=examples, shuffle=True, l_thr=0.5, r_thr=0.75)
            print("data size: ", len(self.task.examples))
        else:
            self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1), semi=semi, shuffle=True)
        self.model = BaseLine(256, 3, 0.2, self.task.max_vocab_indexes['syllable_contents'], 384)
        self.model.to(self.device)
        self.loss_fn = nn.BCELoss()
        self.batch_size = 128
        self.__test_iter = None
        bind_model(self.model)

        if not semi:
            print("Pretrained Loaded")
            nsml.load(checkpoint='59', session='fenneccat/hate_3/644')

    @property
    def test_iter(self) -> Iterator:
        if self.__test_iter:
            self.__test_iter.init_epoch()
            return self.__test_iter
        else:
            self.__test_iter = Iterator(self.task.datasets[1], batch_size=self.batch_size, repeat=False,
                                        sort_key=lambda x: len(x.syllable_contents), train=False,
                                        device=self.device)
            return self.__test_iter

    def train(self):
        print("THIS IS BASELINE TRAINER")
        max_epoch = 15
        optimizer = optim.AdamW(self.model.parameters(),amsgrad=True)
        total_len = len(self.task.datasets[0])
        ds_iter = Iterator(self.task.datasets[0], batch_size=self.batch_size, repeat=False,
                           sort_key=lambda x: len(x.syllable_contents), train=True, device=self.device)
        min_iters = 10
        for epoch in range(max_epoch):
            #print("find break point 1")
            loss_sum, acc_sum, len_batch_sum, f1_sum = 0., 0., 0., 0.
            ds_iter.init_epoch()
            tr_total = math.ceil(total_len / self.batch_size)
            tq_iter = tqdm(enumerate(ds_iter), total=tr_total, miniters=min_iters, unit_scale=self.batch_size,
                           bar_format='{n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}] {desc}', position=0, leave=True)

            self.model.train()
            for i, batch in tq_iter:
                self.model.zero_grad()
                pred = self.model(batch.syllable_contents)
                acc = torch.sum((torch.reshape(pred, [-1]) > 0.5) == (batch.eval_reply > 0.5), dtype=torch.float32)

                #print("find break point 2")
                loss = self.loss_fn(pred, batch.eval_reply)
                #print("find break point 3")
                f1 = f1_score(pred, batch.eval_reply)
                #print("find break point 4")
                loss.backward()
                optimizer.step()

                len_batch = len(batch)
                len_batch_sum += len_batch
                acc_sum += acc.tolist()
                f1_sum += f1.tolist() * len_batch
                loss_sum += loss.tolist() * len_batch
                if i % min_iters == 0:
                    pass
                    tq_iter.set_description('{:2} loss: {:.5}, acc: {:.5}, f1_score: {:.5}'.format(epoch, loss_sum / len_batch_sum, acc_sum / len_batch_sum, f1_sum/len_batch_sum), True)
                if i == 3000:
                    break

            tq_iter.set_description('{:2} loss: {:.5}, acc: {:.5}, f1_score: {:.5}'.format(epoch, loss_sum / total_len, acc_sum / total_len, f1_sum/len_batch_sum), True)

            print(json.dumps(
                {'type': 'train', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_sum / total_len, 'acc': acc_sum / total_len, 'f1_score': f1_sum / total_len}))

            pred_lst, loss_avg, acc_lst, f1_avg, te_total, TP, TN, FP, FN = self.eval(self.test_iter, len(self.task.datasets[1]))
            #print(acc_lst)
            print(json.dumps(
                {'type': 'test', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_avg,  'acc': sum(acc_lst) / te_total, 'f1_score': f1_avg}))

            print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))

            nsml.save(epoch)
            self.save_model(self.model, 'e{}'.format(epoch))

    def eval(self, iter:Iterator, total:int) -> (List[float], float, List[float], int):
        tq_iter = tqdm(enumerate(iter), total=math.ceil(total / self.batch_size),
                       unit_scale=self.batch_size, bar_format='{r_bar}', position=0, leave=True)
        pred_lst = list()
        loss_sum= 0.
        acc_lst = list()
        f1_sum = 0.

        self.model.eval()
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        TP, TN, FP, FN = 0, 0, 0, 0
        for i, batch in tq_iter:
            preds = self.model(batch.syllable_contents)

            accs = torch.eq(preds > 0.5, batch.eval_reply > 0.5).to(torch.float)
            losses = self.loss_fn(preds, batch.eval_reply)
            f1 = f1_score(preds, batch.eval_reply)
            pred_lst += preds.tolist()
            #pred_lst += pred_classes.tolist()
            acc_lst += accs.tolist()
            loss_sum += losses.tolist() * len(batch)
            f1_sum += f1.tolist() * len(batch)
            #print(batch.eval_reply.shape[0])

            for i in range(batch.eval_reply.shape[0]):
                label = batch.eval_reply[i].to(torch.int64)
                prediction = (preds[i]> 0.5).to(torch.int64)
                if prediction == 1 and label == 1:
                    TP += 1
                elif prediction == 1 and label == 0:
                    FP += 1
                elif prediction == 0 and label == 0:
                    TN += 1
                elif prediction == 0 and label == 1:
                    FN += 1

        return pred_lst, loss_sum / total, acc_lst, f1_sum / total, total, TP, TN, FP, FN

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
        task = HateSpeech()
        model = BaseLine(256, 3, 0.2, task.max_vocab_indexes['syllable_contents'], 384)
        model.to("cuda")
        bind_model(model)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        trainer = Trainer(device='cuda')
        trainer.train()
    if args.mode == 'semi':
        print('semi mode!')
        trainer = Trainer(device='cuda', semi = True)
        trainer.train()