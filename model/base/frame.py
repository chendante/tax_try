import torch
import torch.nn as nn
import numpy as np
from torchtext import vocab
from typing import List, Dict, Set


class Frame(nn.Module):
    def __init__(self, inner_module: torch.nn.Module, pretrained_embedding: vocab.Vectors, nn_embedding: nn.Embedding):
        super(Frame, self).__init__()
        self._pretrained_embedding = pretrained_embedding
        self.embedding = nn_embedding
        self.inner_module = inner_module

    def forward(self, query_id, candidate_hyper_ids):
        """
        :param query_id:
        :param candidate_hyper_ids:
        :return:
        """
        hyper_vectors = self.embedding(candidate_hyper_ids)
        query_vectors = self.embedding(query_id)
        return self.inner_module(query_vectors, hyper_vectors)

    @classmethod
    def from_pretrained_embedding(cls, embedding_file_path, inner_module_init_func, cache_path="../data/cache"):
        word_embeddings = vocab.Vectors(embedding_file_path, cache=cache_path)
        padding_idx, embedding_length = word_embeddings.vectors.shape
        padding_embedding = np.zeros(embedding_length)
        inner_module = inner_module_init_func(embedding_length)
        torch_embedding = torch.from_numpy(np.row_stack((word_embeddings.vectors, padding_embedding)))
        nn_embedding = nn.Embedding.from_pretrained(embeddings=torch_embedding, freeze=False, padding_idx=padding_idx)
        return cls(inner_module, pretrained_embedding=word_embeddings, nn_embedding=nn_embedding)

    @property
    def padding_idx(self):
        return self.embedding.padding_idx

    def get_word_idx(self, word: str):
        if word in self._pretrained_embedding.stoi:
            return self._pretrained_embedding.stoi[word]
        return self.padding_idx

    @staticmethod
    def loss_fct(logits: torch.Tensor, targets: torch.Tensor, attention_mask: torch.Tensor):  # TODO: 可能应该根据权重来计算
        fuc = torch.nn.BCELoss()
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1)
        active_targets = torch.where(
            active_loss, targets.view(-1), active_logits
        )
        loss = fuc(active_logits, active_targets)
        return loss
