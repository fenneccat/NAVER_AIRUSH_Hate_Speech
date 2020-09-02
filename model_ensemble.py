from torch import nn
import torch
import torch.nn.functional as F

class BaseLine(nn.Module):
    def __init__(self, hidden_dim, filter_size, dropout_rate, vocab_size, embedding_dim, pre_trained_embedding=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim

        if pre_trained_embedding is None:
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(pre_trained_embedding, freeze=False, padding_idx=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv1d = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.filter_size)
        self.bi_rnn = nn.LSTM(self.hidden_dim, int(self.hidden_dim / 2), batch_first=True, bidirectional=True)
        self.uni_rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.max_pool = nn.AdaptiveAvgPool2d((1, self.hidden_dim))
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).transpose(0, 1).transpose(1, 2)
        x = self.conv1d(x).transpose(1, 2).transpose(0, 1)
        x = self.relu(x)
        x = self.dropout(x)
        x_res = x
        x, _ = self.bi_rnn(x)
        x, _ = self.uni_rnn(x + x_res)
        x = self.dropout(x)
        x, _ = torch.max(x, 0)
        x = self.linear(x)
        x = self.sigmoid(x).squeeze()
        return x

class MyEnsemble(nn.Module):
    def __init__(self, models):
        super(MyEnsemble, self).__init__()
        self.models = models
        self.sigmoid = nn.Sigmoid()

        # Create new classifier
        self.f_linear = nn.Linear(len(self.models), 1)

    def forward(self, x):
        xx = []
        for model in self.models:
            xx.append(model(x.clone()).reshape(-1,1))

        x = torch.cat(xx, dim=1)
        x = self.f_linear(x)
        x = self.sigmoid(x).squeeze()

        return x

class MyEnsemble_mean(nn.Module):
    def __init__(self, models):
        super(MyEnsemble_mean, self).__init__()
        self.models = models
        self.sigmoid = nn.Sigmoid()

        # Create new classifier
        self.f_linear = nn.Linear(len(self.models), 1)

    def forward(self, x):
        xx = []
        for model in self.models:
            xx.append(model(x.clone()).reshape(-1,1))

        x = torch.cat(xx, dim=1)
        x = x.mean(dim=1)

        return x

class MyEnsemble_majority(nn.Module):
    def __init__(self, models):
        super(MyEnsemble_majority, self).__init__()
        self.models = models
        self.sigmoid = nn.Sigmoid()

        # Create new classifier
        self.f_linear = nn.Linear(len(self.models), 1)

    def forward(self, x):
        xx = []
        for model in self.models:
            xx.append((model(x.clone()) > 0.5).to(torch.float32).reshape(-1,1))

        x = torch.cat(xx, dim=1)
        x = x.mean(dim=1)

        return x
