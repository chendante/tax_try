import math

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter


class Crim(nn.Module):
    def __init__(self, embedding_length, k=24, query_vectors=None):
        super(Crim, self).__init__()
        self._k = k
        self._embedding_length = embedding_length
        self._criterion = nn.BCELoss()

        self.weight = Parameter(torch.FloatTensor(embedding_length, k * embedding_length))
        self.Linear = torch.nn.Linear(k, 1, True)
        # self.weight_init(query_vectors)
        self.reset_parameters()

    def weight_init(self, query_vectors):
        count = 0
        for i in range(self.weight.shape[0]):
            for j in range(self.weight.shape[1]):
                self.weight[i][j] = torch.FloatTensor(query_vectors[count])
                count += 1
                if count == query_vectors.shape[0]:
                    count = 0
        self.weight = Parameter(self.weight + (0.01 ** 0.5) * torch.randn(self.weight.shape), requires_grad=True)

    def reset_parameters(self):
        nn.init.kaiming_uniform(self.weight, a=0)

    def forward(self, e_qs: torch.Tensor, e_hs: torch.Tensor):
        """
        :param e_qs: embeddings of hyponyms; shape: batch_size * embedding_length
        :param e_hs: a group embeddings of candidate hypernyms; shape: batch_size * group_size * embedding_length
        :return: ys: each number is the prob of a candidate hypernym; shape: batch_size * group_size
        """
        p = torch.mm(e_qs, self.weight).reshape(e_qs.shape[0], self._k, self._embedding_length).transpose(1, 2)
        s = torch.bmm(e_hs, p)

        ys = torch.sigmoid(self.Linear(s).squeeze())
        return ys

    def loss_function(self, output, target):
        return self._criterion(output, target)

