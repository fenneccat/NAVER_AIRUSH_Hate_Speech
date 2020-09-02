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

from model_ensemble import BaseLine, MyEnsemble
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
        examples = HateSpeech(raw_data, shuffle = False).examples
        tensors = [torch.tensor(ex.syllable_contents, device='cuda').reshape([-1, 1]) for ex in examples]
        results = [model(ex).tolist() for ex in tensors]
        return results

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

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        # f1 = f1.clamp(min=epsilon, max=1 - epsilon)
        # f1.requires_grad = is_training
        return f1.mean()

class Trainer(object):
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)

    def __init__(self, hdfs_host: str = None, device: str = 'cpu', num_models = 2):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1))
        self.models = []
        for i in range(num_models):
            self.models.append(BaseLine(256, 3, 0.2, self.task.max_vocab_indexes['syllable_contents'], 384))
            self.models[-1].to(self.device)
        self.model = MyEnsemble(self.models)
        self.model.to(self.device)
        self.loss_fn = nn.BCELoss()
        #self.loss_fn = CB_loss
        self.batch_size = 128
        self.__test_iter = None
        bind_model(self.models[0])  # non-shuffle
        nsml.load(checkpoint='8', session='fenneccat/hate_3/64')
        bind_model(self.models[1])  # shuffle 1
        nsml.load(checkpoint='4', session='fenneccat/hate_3/72')
        bind_model(self.models[2])  # shuffle 2
        nsml.load(checkpoint='2', session='fenneccat/hate_3/82')
        bind_model(self.models[3])  # shuffle 3
        nsml.load(checkpoint='11', session='fenneccat/hate_3/78')
        bind_model(self.models[4])  # shuffle 4
        nsml.load(checkpoint='6', session='fenneccat/hate_3/81')
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

        # Freeze these models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad_(False)

        bind_model(self.model)

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
            loss_sum, acc_sum, len_batch_sum, f1_sum = 0., 0., 0., 0.
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
                f1 = f1_score(pred, batch.eval_reply)
                #loss = self.loss_fn(pred, batch.eval_reply, [70181, 29819], 2, *self.loss_param)
                loss.backward()
                optimizer.step()

                len_batch = len(batch)
                len_batch_sum += len_batch
                acc_sum += acc.tolist()
                f1_sum += f1.tolist() * len_batch
                loss_sum += loss.tolist() * len_batch
                if i % min_iters == 0:
                    tq_iter.set_description('{:2} loss: {:.5}, acc: {:.5}, f1_score: {:.5}'.format(epoch, loss_sum / len_batch_sum, acc_sum / len_batch_sum, f1_sum/len_batch_sum), True)
                if i == 3000:
                    break

            tq_iter.set_description('{:2} loss: {:.5}, acc: {:.5}, f1_score: {:.5}'.format(epoch, loss_sum / total_len, acc_sum / total_len, f1_sum/len_batch_sum), True)

            print(json.dumps(
                {'type': 'train', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_sum / total_len, 'acc': acc_sum / total_len,
                 'f1_score': f1_sum / total_len}))

            pred_lst, loss_avg, acc_lst, f1_avg, te_total, TP, TN, FP, FN = self.eval(self.test_iter, len(self.task.datasets[1]))

            print(json.dumps(
                {'type': 'test', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_avg, 'acc': sum(acc_lst) / te_total, 'f1_score': f1_avg}))
            print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
            nsml.save(epoch)
            self.save_model(self.model, 'e{}'.format(epoch))

    def eval(self, iter:Iterator, total:int) -> (List[float], float, List[float], int):
        tq_iter = tqdm(enumerate(iter), total=math.ceil(total / self.batch_size),
                       unit_scale=self.batch_size, bar_format='{r_bar}')
        pred_lst = list()
        loss_sum= 0.
        f1_sum = 0.
        acc_lst = list()
        TP, TN, FP, FN = 0, 0, 0, 0

        self.model.eval()
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        for i, batch in tq_iter:
            preds = self.model(batch.syllable_contents)
            #_, pred_classes = preds.topk(1, dim=1)

            #accs = torch.eq(pred_classes, batch.eval_reply.view(*pred_classes.shape)).to(torch.float).squeeze()
            #print("accs")
            #print(accs)

            #losses = self.loss_fn(preds, batch.eval_reply, [70181, 29819], 2, *self.loss_param)

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
                prediction = torch.round(preds[i]).to(torch.int64)
                if prediction == 1 and label == 1:
                    TP += 1
                elif prediction == 1 and label == 0:
                    FP += 1
                elif prediction == 0 and label == 0:
                    TN += 1
                elif prediction == 0 and label == 1:
                    FN += 1
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
        models = []
        num_models = 26
        for i in range(num_models):
            models.append(BaseLine(256, 3, 0.2, task.max_vocab_indexes['syllable_contents'], 384))
            models[-1].to("cuda")

        bind_model(models[0])  # non-shuffle
        nsml.load(checkpoint='8', session='fenneccat/hate_3/64') #0.8759996676445008
        bind_model(models[1])  # shuffle 1
        nsml.load(checkpoint='4', session='fenneccat/hate_3/72') #0.8775300717353821
        bind_model(models[2])  # shuffle 2
        nsml.load(checkpoint='2', session='fenneccat/hate_3/82') #0.8654762921333313
        bind_model(models[3])  # shuffle 3
        nsml.load(checkpoint='11', session='fenneccat/hate_3/78') #0.8675199780464172
        bind_model(models[4])  # shuffle 4
        nsml.load(checkpoint='6', session='fenneccat/hate_3/81') #0.8845085878372192
        bind_model(models[5])  # non-shuffle, hate_2
        nsml.load(checkpoint='6', session='fenneccat/hate_2/131')
        bind_model(models[6])  # shuffle 1, hate_2
        nsml.load(checkpoint='23', session='fenneccat/hate_2/96')
        bind_model(models[7])  # shuffle 2, hate_2
        nsml.load(checkpoint='4', session='fenneccat/hate_2/97')
        bind_model(models[8])  # shuffle 3, hate_2
        nsml.load(checkpoint='5', session='fenneccat/hate_2/98')
        bind_model(models[9])  # shuffle 4, hate_2
        nsml.load(checkpoint='6', session='fenneccat/hate_2/130')## ======= 여기까지 모델 메모리 정리함
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
        nsml.load(checkpoint='11', session='fenneccat/hate_3/544')
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

        model = MyEnsemble(models)
        model.to("cuda")
        bind_model(model)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        trainer = Trainer(device='cuda', num_models=26)
        trainer.train()