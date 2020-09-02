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
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertConfig

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

        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model']

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        #model.load_state_dict(checkpoint['model'])

    def infer(raw_data, **kwargs):
        model.eval()
        examples = HateSpeech(raw_data).examples
        tensors = [torch.tensor(ex.syllable_contents, device='cuda').reshape([-1, 1]) for ex in examples]

        results = []
        for ex in tensors:
            outputs = model(ex.transpose(0,1))
            #print('ex', ex)

            logit = outputs[0]
            #print("single ex: ", logit)

            _, pred = logit.topk(1, dim=1)
            results.append(pred.squeeze().tolist())

        return results

    nsml.bind(save=save, load=load, infer=infer)

class BERTTrainer(object):
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)

    def __init__(self, hdfs_host: str = None, device: str = 'cpu'):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1))

        self.configuration = BertConfig(vocab_size=self.task.max_vocab_indexes['syllable_contents'], hidden_dropout_prob = 0.1,
                                        num_hidden_layers=2)

        self.model = BertForSequenceClassification(self.configuration)
        #TODO: pretrained model load 해야함
        #self.model.bert.load_state_dict(torch.load(PATH))

        #bind_model(self.model.bert)

        self.model.to(self.device)
        self.loss_fn = nn.BCELoss()
        self.batch_size = 128
        self.__test_iter = None
        bind_model(self.model)
        nsml.load(checkpoint='27', session='fenneccat/hate_raw/38')

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
        print("THIS IS BERT Classifier TRAINER")
        max_epoch = 32
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        total_len = len(self.task.datasets[0])
        ds_iter = Iterator(self.task.datasets[0], batch_size=self.batch_size, repeat=False,
                           sort_key=lambda x: len(x.syllable_contents), train=True, device=self.device)
        min_iters = 10
        for epoch in range(max_epoch):
            loss_sum, acc_sum, len_batch_sum = 0., 0., 0.
            ds_iter.init_epoch()
            tr_total = math.ceil(total_len / self.batch_size)
            tq_iter = tqdm(enumerate(ds_iter), total=tr_total, miniters=min_iters, unit_scale=self.batch_size,
                           bar_format='{n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}] {desc}')

            self.model.train()
            for i, batch in tq_iter:
                self.model.zero_grad()
                outputs = self.model(batch.syllable_contents.transpose(0,1), labels=batch.eval_reply.to(torch.long))
                loss = outputs[0]
                logit = outputs[1]

                _, preds = logit.topk(1, dim=1)

                #print("[train] pred shape", preds.shape)
                #print("[train] pred", preds)

                acc = torch.sum(preds == batch.eval_reply.view(*preds.shape), dtype=torch.float32)

                loss.backward()
                optimizer.step()

                len_batch = len(batch)
                len_batch_sum += len_batch
                acc_sum += acc.tolist()
                loss_sum += loss.tolist() * len_batch
                if i % min_iters == 0:
                    tq_iter.set_description('{:2} loss: {:.5}, acc: {:.5}'.format(epoch, loss_sum / len_batch_sum, acc_sum / len_batch_sum), True)
                if i == 3000:
                    break

            tq_iter.set_description('{:2} loss: {:.5}, acc: {:.5}'.format(epoch, loss_sum / total_len, acc_sum / total_len), True)

            print(json.dumps(
                {'type': 'train', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_sum / total_len, 'acc': acc_sum / total_len}))

            pred_lst, loss_avg, acc_lst, te_total = self.eval(self.test_iter, len(self.task.datasets[1]))
            #print(acc_lst)

            print(json.dumps(
                {'type': 'test', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_avg,  'acc': sum(acc_lst) / te_total}))
            nsml.save(epoch)
            self.save_model(self.model, 'e{}'.format(epoch))

    def eval(self, iter:Iterator, total:int) -> (List[float], float, List[float], int):
        tq_iter = tqdm(enumerate(iter), total=math.ceil(total / self.batch_size),
                       unit_scale=self.batch_size, bar_format='{r_bar}')
        pred_lst = list()
        loss_sum= 0.
        acc_lst = list()

        self.model.eval()
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        for i, batch in tq_iter:

            outputs = self.model(batch.syllable_contents.transpose(0,1), labels=batch.eval_reply.to(torch.long))
            losses = outputs[0]
            logits = outputs[1]

            _, preds = logits.topk(1, dim=1)

            #print("[test] pred shape: ", preds.shape)
            #print("[test] pred: ", preds)

            accs = torch.eq(preds, batch.eval_reply.view(*preds.shape)).to(torch.float)

            pred_lst += preds.tolist()
            accs = accs.squeeze()
            acc_lst += accs.tolist()
            loss_sum += losses.tolist() * len(batch)

            '''
            for i in range(batch.eval_reply.shape[0]):
                label = batch.eval_reply[i].to(torch.int64)
                class_correct[label] += accs[i].item()
                class_total[label] += 1
            '''

        '''
        for i in range(2):
            if class_total[i] > 0:
                print('Test Accuracy of label %d: %2d%% (%2d/%2d)' % (
                    i, 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of label %d: N/A (no training examples)' % (i))
        '''

        return pred_lst, loss_sum / total, acc_lst, total

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
        configuration = BertConfig(vocab_size=task.max_vocab_indexes['syllable_contents'], hidden_dropout_prob = 0.1
                                   , num_hidden_layers=2)

        model = BertForSequenceClassification(configuration)
        model.to("cuda")
        bind_model(model)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        trainer = BERTTrainer(device='cuda')
        trainer.train()