import torch
from typing import *
import model
from torch.utils.data import DataLoader, TensorDataset


class Sampler:
    def __init__(self, data_words: List[str], gold_words_list: List[List[str]], frame: model.Frame, batch_size,
                 padding_max=256):
        self._batch_size = batch_size
        self._padding_max = padding_max
        self._frame = frame
        self._tax_graph = model.tax.TaxStruct(gold_words_list, threshold=1.0, num_bar=3, add_num_bar=10)
        self._gold_words_list = gold_words_list
        self._data_words = data_words
        self.data_loader = self._init_data_loader()

    def _init_data_loader(self):
        """
            query_id_list: [query_id...]
            hyper_ids_list: [[hyper_id...]...]
            target_list: [[1,0...]...]
        :return:
        query, hypers, attention_mask, target
        """
        query_id_list = []
        hyper_ids_list = []
        targets_list = []
        attention_masks_list = []
        for hypo, hypers in zip(self._data_words, self._gold_words_list):
            hyper_set = set([hyper for hyper in hypers if self._tax_graph.has_node(hyper)])  # 获取所有在tax_g中的terms
            while len(hyper_set) > 0:
                hyper = hyper_set.pop()
                bros = self._tax_graph.get_bros(hyper)
                hyper_ids = [self._frame.get_word_idx(bro) for bro in bros]
                hyper_ids.extend([self._frame.padding_idx] * (self._padding_max - len(bros)))
                targets = []
                for bro in bros:
                    if bro in hyper_set:
                        targets.append(1)
                    else:
                        targets.append(0)
                query_id_list.append(self._frame.get_word_idx(hypo))
                hyper_ids_list.append(hyper_ids)
                targets_list.append(targets)
                attention_masks_list.append(([1] * len(bros)).extend([0] * (self._padding_max - len(bros))))
        # 转torch tensor
        query_ids = torch.Tensor(query_id_list)
        hyper_ids = torch.Tensor(hyper_ids_list)
        targets = torch.Tensor(targets_list)
        attention_masks = torch.Tensor(attention_masks_list)
        # 转data loader
        dataset = TensorDataset(query_ids, hyper_ids, attention_masks, targets)
        data_loader = DataLoader(dataset=dataset, batch_size=self._batch_size, shuffle=True)
        return data_loader

    def __iter__(self):
        return self.data_loader.__iter__()

    def __len__(self):
        return self.data_loader.__len__()
