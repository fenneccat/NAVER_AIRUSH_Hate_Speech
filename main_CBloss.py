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
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)

    def __init__(self, hdfs_host: str = None, device: str = 'cpu'):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1))
        self.model = BaseLine(256, 3, 0.2, self.task.max_vocab_indexes['syllable_contents'], 384)
        self.model.to(self.device)
        self.loss_fn = nn.BCELoss()
        #self.loss_fn = CB_loss
        self.batch_size = 128
        self.__test_iter = None
        self.loss_param = ["softmax", 0.9999, 2.0]
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
        print("THIS IS BASELINE TRAINER")
        max_epoch = 100
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

                #print("[train] pred shape: ", pred.shape)
                #print("[train] pred: ", pred)

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
            preds = self.model(batch.syllable_contents)
            #_, pred_classes = preds.topk(1, dim=1)

            #print("[test] pred shape: ", preds.shape)
            #print("[test] pred: ", preds)

            #accs = torch.eq(pred_classes, batch.eval_reply.view(*pred_classes.shape)).to(torch.float).squeeze()
            #print("accs")
            #print(accs)

            #losses = self.loss_fn(preds, batch.eval_reply, [70181, 29819], 2, *self.loss_param)

            accs = torch.eq(preds > 0.5, batch.eval_reply > 0.5).to(torch.float)
            losses = self.loss_fn(preds, batch.eval_reply)
            pred_lst += preds.tolist()
            #pred_lst += pred_classes.tolist()
            acc_lst += accs.tolist()
            loss_sum += losses.tolist() * len(batch)
            #print(batch.eval_reply.shape[0])
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
        model = BaseLine(256, 3, 0.2, task.max_vocab_indexes['syllable_contents'], 384)
        model.to("cuda")
        bind_model(model)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        trainer = Trainer(device='cuda')
        trainer.train()



'''


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(logits, labels, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      logits: A float tensor of size [batch, no_of_classes].
      labels: A int tensor of size [batch].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels.to(torch.int64), no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1)
    weights = torch.mul(weights.cuda(),labels_one_hot)
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)

    return cb_loss


class Trainer_CBLoss(object):
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)

    def __init__(self, hdfs_host: str = None, device: str = 'cpu'):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1))
        self.model = BaseLine_two_class(256, 3, 0.2, self.task.max_vocab_indexes['syllable_contents'], 384)
        self.model.to(self.device)
        #self.loss_fn = nn.BCELoss()
        self.loss_fn = CB_loss
        self.batch_size = 128
        self.__test_iter = None
        self.loss_param = ["softmax", 0.9999, 2.0]
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
        print("THIS IS CB LOSS TRAININER")
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
                #acc = torch.sum((torch.reshape(pred, [-1]) > 0.5) == (batch.eval_reply > 0.5), dtype=torch.float32)
                ## two label pred accuracy
                _, pred_class = pred.topk(1, dim=1)

                acc = torch.sum(pred_class == batch.eval_reply.view(*pred_class.shape))

                #labels_one_hot = F.one_hot(batch.eval_reply.to(torch.int64), 2).float()
                #loss = self.loss_fn(pred, labels_one_hot)
                #loss = self.loss_fn(pred, batch.eval_reply)
                loss = self.loss_fn(pred, batch.eval_reply, [20, 10], 2, *self.loss_param)
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
            preds = self.model(batch.syllable_contents)
            _, pred_classes = preds.topk(1, dim=1)

            accs = torch.eq(pred_classes, batch.eval_reply.view(*pred_classes.shape)).to(torch.float).squeeze()
            #print("accs")
            #print(accs)

            losses = self.loss_fn(preds, batch.eval_reply, [20, 10], 2, *self.loss_param)

            #accs = torch.eq(preds > 0.5, batch.eval_reply > 0.5).to(torch.float)
            #losses = self.loss_fn(preds, batch.eval_reply)
            #pred_lst += preds.tolist()
            pred_lst += pred_classes.tolist()
            acc_lst += accs.tolist()
            loss_sum += losses.tolist() * len(batch)
            
            for i in range(batch.eval_reply.shape[0]):
                label = batch.eval_reply[i].to(torch.int64)
                class_correct[label] += accs[i].item()
                class_total[label] += 1
            
        
        for i in range(2):
            if class_total[i] > 0:
                print('Test Accuracy of label %d: %2d%% (%2d/%2d)' % (
                    i, 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of label %d: N/A (no training examples)' % (i))
        

        return pred_lst, loss_sum / total, acc_lst, total

    def save_model(self, model, appendix=None):
        file_name = 'model'
        if appendix:
            file_name += '_{}'.format(appendix)
        torch.save({'model': model, 'task': type(self.task).__name__}, file_name)

'''