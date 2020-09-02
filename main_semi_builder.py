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

import torch.nn.functional as F

from model_ensemble import BaseLine,MyEnsemble
from model_ensemble import BaseLine, MyEnsemble_majority, MyEnsemble_mean
from data import HateSpeech

def bind_model(model):
    def save(dirname, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))

    def load(dirname, *args):
        #print("LOAD!!")
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        #print(checkpoint['model'])
        model.load_state_dict(checkpoint['model'])

    def infer(raw_data, **kwargs):
        model.eval()
        examples = HateSpeech(raw_data, test = True).examples
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
            examples = json.load(json_file)

    def infer(raw_data, **kwargs):
        pass

    nsml.bind(save=save, load=load, infer=infer)

class Trainer(object):
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATehH[0])
    RAW_DATA_PATH = '{}/train/raw.json'.format(DATASET_PATH[1])

    def __init__(self, hdfs_host: str = None, device: str = 'cpu', num_models = 2):
        self.device = device
        self.task = HateSpeech(self.RAW_DATA_PATH, raw=True)
        self.models = []
        for i in range(num_models):
            self.models.append(BaseLine(256, 3, 0.2, self.task.max_vocab_indexes['syllable_contents'], 384))
            self.models[-1].to(self.device)
        self.model = MyEnsemble_mean(self.models)
        self.model.to(self.device)
        self.loss_fn = nn.BCELoss()
        self.batch_size = 128
        self.__test_iter = None
        bind_model(self.models[0])  # non-shuffle
        nsml.load(checkpoint='8', session='fenneccat/hate_3/64')  # 0.8759996676445008
        bind_model(self.models[1])  # shuffle 1
        nsml.load(checkpoint='4', session='fenneccat/hate_3/72')  # 0.8775300717353821
        bind_model(self.models[2])  # shuffle 2
        nsml.load(checkpoint='2', session='fenneccat/hate_3/82')  # 0.8654762921333313
        bind_model(self.models[3])  # shuffle 3
        nsml.load(checkpoint='11', session='fenneccat/hate_3/78')  # 0.8675199780464172
        bind_model(self.models[4])  # shuffle 4
        nsml.load(checkpoint='6', session='fenneccat/hate_3/81')  # 0.8845085878372192
        bind_model(self.models[5])  # non-shuffle, hate_2
        nsml.load(checkpoint='6', session='fenneccat/hate_2/131')
        bind_model(self.models[6])  # shuffle 1, hate_2
        nsml.load(checkpoint='23', session='fenneccat/hate_2/96')
        bind_model(self.models[7])  # shuffle 2, hate_2
        nsml.load(checkpoint='4', session='fenneccat/hate_2/97')
        bind_model(self.models[8])  # shuffle 3, hate_2
        nsml.load(checkpoint='5', session='fenneccat/hate_2/98')
        bind_model(self.models[9])  # shuffle 4, hate_2
        nsml.load(checkpoint='6', session='fenneccat/hate_2/130')
        bind_model(self.models[10])  # shuffle 5
        nsml.load(checkpoint='10', session='fenneccat/hate_3/471')
        bind_model(self.models[11])  # shuffle 6
        nsml.load(checkpoint='4', session='fenneccat/hate_3/472')
        bind_model(self.models[12])  # shuffle 7
        nsml.load(checkpoint='8', session='fenneccat/hate_3/476')
        bind_model(self.models[13])  # shuffle 8
        nsml.load(checkpoint='3', session='fenneccat/hate_3/477')
        bind_model(self.models[14])  # shuffle 9
        nsml.load(checkpoint='3', session='fenneccat/hate_3/501')
        bind_model(self.models[15])  # shuffle 10
        nsml.load(checkpoint='8', session='fenneccat/hate_3/503')
        bind_model(self.models[16])  # shuffle 11
        nsml.load(checkpoint='6', session='fenneccat/hate_3/510')
        bind_model(self.models[17])  # shuffle 12
        nsml.load(checkpoint='8', session='fenneccat/hate_3/542')
        bind_model(self.models[18])  # shuffle 13
        nsml.load(checkpoint='4', session='fenneccat/hate_3/543')
        bind_model(self.models[19])  # shuffle 14
        nsml.load(checkpoint='11', session='fenneccat/hate_3/544')  ## ======= 여기까지 모델 메모리 정리함
        bind_model(self.models[20])  # shuffle 15
        nsml.load(checkpoint='13', session='fenneccat/hate_3/545')
        bind_model(self.models[21])  # shuffle 16
        nsml.load(checkpoint='6', session='fenneccat/hate_3/547')
        bind_model(self.models[22])  # shuffle 17
        nsml.load(checkpoint='6', session='fenneccat/hate_3/548')
        bind_model(self.models[23])  # shuffle 18
        nsml.load(checkpoint='7', session='fenneccat/hate_3/549')
        bind_model(self.models[24])  # shuffle 19
        nsml.load(checkpoint='4', session='fenneccat/hate_3/550')
        bind_model(self.models[25])  # shuffle 20
        nsml.load(checkpoint='3', session='fenneccat/hate_3/551')

        # Freeze these models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad_(False)

        for param in self.model.parameters():
            param.requires_grad_(False)

        bind_model(self.model)
        nsml.load(checkpoint='0', session='fenneccat/hate_3/577')

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
        print("THIS IS BASELINE ENSEMBLE TRAINER")
        max_epoch = 32
        optimizer = optim.Adam(self.model.parameters())
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
                pred = self.model(batch.syllable_contents)
                ## one sigmoid pred accuracy
                acc = torch.sum((torch.reshape(pred, [-1]) > 0.5) == (batch.eval_reply > 0.5), dtype=torch.float32)
                ## two label pred accuracy
                #_, pred_class = pred.topk(1, dim=1)

                #acc = torch.sum(pred_class == batch.eval_reply.view(*pred_class.shape))

                #labels_one_hot = F.one_hot(batch.eval_reply.to(torch.int64), 2).float()
                #loss = self.loss_fn(pred, labels_one_hot)
                loss = self.loss_fn(pred, batch.eval_reply)
                #loss = self.loss_fn(pred, batch.eval_reply, [70181, 29819], 2, *self.loss_param)
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

    def eval(self):
        print("THIS IS BASELINE ENSEMBLE TRAINER EVAL")

        semione, semizero = 0, 0

        origin = HateSpeech(self.TRAIN_DATA_PATH, copy=True)
        examples = origin.examples
        print("LOAD hate_2 Complete!")

        total_len = len(self.task.datasets[0])
        ds_iter = Iterator(self.task.datasets[0], batch_size=self.batch_size, repeat=False,
                           sort_key=lambda x: len(x.syllable_contents), train=False, device=self.device)
        raw_total = math.ceil(total_len / self.batch_size)
        print(self.task.fields)
        min_iters = 10
        raw_iter = tqdm(enumerate(ds_iter), total=raw_total, miniters=min_iters, unit_scale=self.batch_size)

        self.model.eval()
        preds_all = []
        for i, batch in raw_iter:
            if i % 1000 == 0: print("batch num/total batchs: {}/{}".format(i,len(raw_iter)))
            preds = self.model(batch.syllable_contents)
            preds_all.extend(preds.tolist())

            for i in range(batch.syllable_contents.shape[1]):
                example_dict = dict()
                example_dict['syllable_contents'] = batch.syllable_contents.transpose(0,1)[i].tolist()
                example_dict['prediction_score'] = preds[i].tolist()
                example_dict['eval_reply'] = int(preds[i] > 0.5)
                examples.append(example_dict)


        def count_range(list1, l, r):
            c = 0
            # traverse in the list1
            for x in list1:
                # condition check
                if x >= l and x < r:
                    c += 1
            return c

        # driver code
        for xx in range(0, 100, 5):
            x = xx/100
            print("{}~{}: {}".format(x, x+0.05,count_range(preds_all, x, x+0.05)))

        print("ADD PSEUDO LABEL DONE!!")

        bind_json(examples)
        nsml.save('raw_data')

        print("file saved!!")


    def save_model(self, model, appendix=None):
        file_name = 'model'
        if appendix:
            file_name += '_{}'.format(appendix)
        torch.save({'model': model, 'task': type(self.task).__name__}, file_name)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', default='semi')
    parser.add_argument('--pause', default=0)
    args = parser.parse_args()
    if args.pause:
        task = HateSpeech()
        models = []
        num_models = 26
        for i in range(num_models):
            models.append(BaseLine(256, 3, 0.2, task.max_vocab_indexes['syllable_contents'], 384))
            models[-1].to("cuda")

        bind_model(models[0])  # non-shuffle
        nsml.load(checkpoint='8', session='fenneccat/hate_3/64')  # 0.8759996676445008
        bind_model(models[1])  # shuffle 1
        nsml.load(checkpoint='4', session='fenneccat/hate_3/72')  # 0.8775300717353821
        bind_model(models[2])  # shuffle 2
        nsml.load(checkpoint='2', session='fenneccat/hate_3/82')  # 0.8654762921333313
        bind_model(models[3])  # shuffle 3
        nsml.load(checkpoint='11', session='fenneccat/hate_3/78')  # 0.8675199780464172
        bind_model(models[4])  # shuffle 4
        nsml.load(checkpoint='6', session='fenneccat/hate_3/81')  # 0.8845085878372192
        bind_model(models[5])  # non-shuffle, hate_2
        nsml.load(checkpoint='6', session='fenneccat/hate_2/131')
        bind_model(models[6])  # shuffle 1, hate_2
        nsml.load(checkpoint='23', session='fenneccat/hate_2/96')
        bind_model(models[7])  # shuffle 2, hate_2
        nsml.load(checkpoint='4', session='fenneccat/hate_2/97')
        bind_model(models[8])  # shuffle 3, hate_2
        nsml.load(checkpoint='5', session='fenneccat/hate_2/98')
        bind_model(models[9])  # shuffle 4, hate_2
        nsml.load(checkpoint='6', session='fenneccat/hate_2/130')
        bind_model(models[10])  # shuffle 5
        nsml.load(checkpoint='10', session='fenneccat/hate_3/471')
        bind_model(models[11])  # shuffle 6
        nsml.load(checkpoint='4', session='fenneccat/hate_3/472')
        bind_model(models[12])  # shuffle 7
        nsml.load(checkpoint='8', session='fenneccat/hate_3/476')
        bind_model(models[13])  # shuffle 8
        nsml.load(checkpoint='3', session='fenneccat/hate_3/477')
        bind_model(models[14])  # shuffle 9
        nsml.load(checkpoint='3', session='fenneccat/hate_3/501')
        bind_model(models[15])  # shuffle 10
        nsml.load(checkpoint='8', session='fenneccat/hate_3/503')
        bind_model(models[16])  # shuffle 11
        nsml.load(checkpoint='6', session='fenneccat/hate_3/510')
        bind_model(models[17])  # shuffle 12
        nsml.load(checkpoint='8', session='fenneccat/hate_3/542')
        bind_model(models[18])  # shuffle 13
        nsml.load(checkpoint='4', session='fenneccat/hate_3/543')
        bind_model(models[19])  # shuffle 14
        nsml.load(checkpoint='11', session='fenneccat/hate_3/544')  ## ======= 여기까지 모델 메모리 정리함
        bind_model(models[20])  # shuffle 15
        nsml.load(checkpoint='13', session='fenneccat/hate_3/545')
        bind_model(models[21])  # shuffle 16
        nsml.load(checkpoint='6', session='fenneccat/hate_3/547')
        bind_model(models[22])  # shuffle 17
        nsml.load(checkpoint='6', session='fenneccat/hate_3/548')
        bind_model(models[23])  # shuffle 18
        nsml.load(checkpoint='7', session='fenneccat/hate_3/549')
        bind_model(models[24])  # shuffle 19
        nsml.load(checkpoint='4', session='fenneccat/hate_3/550')
        bind_model(models[25])  # shuffle 20
        nsml.load(checkpoint='3', session='fenneccat/hate_3/551')

        model = MyEnsemble_mean(models)
        model.to("cuda")
        bind_model(model)
        nsml.load(checkpoint='2', session='fenneccat/hate_2/175')
        nsml.paused(scope=locals())def bind_json(examples):
    def save(dirname, *args):
        with open(os.path.join(dirname, 'raw_labeled.json'), "w") as json_file:
            json.dump(examples, json_file)

    def load(dirname, *args):
        with open(os.path.join(dirname, 'raw_labeled.json'), "r") as json_file:
            examples = json.load(json_file)

    def infer(raw_data, **kwargs):
        pass


    if args.mode == 'semi':
        trainer = Trainer(device='cuda', num_models=26)
        trainer.eval()