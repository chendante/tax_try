import torch
from typing import *
import model
from torch.utils.data import DataLoader, Dataset
import networkx as nx
import random
import bpemb


class TaxStruct(nx.DiGraph):
    root = "<root>"

    def __init__(self, edges):
        all_edges = self.get_edges_with_root(edges)
        super().__init__(all_edges)
        self.check_useless_edge()

    @classmethod
    def get_edges_with_root(cls, edges):
        edges_with_root = edges  # hyper to hypo
        children = set([e[1] for e in edges])
        entities = list(set([e[0] for e in edges]).union(children))
        for entity in entities:
            if entity not in children:
                edges_with_root.append((cls.root, entity))
        return edges_with_root

    def check_useless_edge(self):
        """
        删除对于taxonomy结构来说无用的边
        """
        bad_edges = []
        for node in self.nodes:
            if len(self.pred[node]) <= 1:
                continue
            # if self.out_degree(node) == 0:
            # print(node)
            for pre in self.predecessors(node):
                for ppre in self.predecessors(node):
                    if ppre != pre:
                        if nx.has_path(self, pre, ppre):
                            bad_edges.append((pre, node))
                            # print(node, pre, ppre)
        self.remove_edges_from(bad_edges)

    def all_leaf_nodes(self):
        # 根据是否只要单一父节点的叶节点可进行更改
        return [node for node in self.nodes.keys() if self.out_degree(node) == 0 and self.in_degree(node) == 1]

    def all_paths(self):
        paths = []
        for node in self.nodes:
            path = nx.shortest_path(self, source=self.root, target=node)
            if len(path) > 2:
                paths.append(path)
        return paths

    def get_margin(self, path, node):
        m_path = nx.shortest_path(self, self.root, node)
        return len(path) + len(m_path) - len(set(path).union(set(m_path)))


class Sampler(Dataset):
    def __init__(self, edges, tokenizer: bpemb.BPEmb, padding_max=256):
        """
        :param edges: (hyper, hypo)
        :param tokenizer:
        :param padding_max:
        """
        self._margins = []
        self._pos_paths = []
        self._neg_paths = []
        self._padding_max = padding_max
        self._tokenizer = tokenizer
        self._padding_id = tokenizer.emb.vocab.get("<pad>").index
        self._tax_graph = TaxStruct(edges)
        leaf_nodes = self._tax_graph.all_leaf_nodes()
        random.seed(0)
        testing_nodes = random.sample(leaf_nodes, int(len(leaf_nodes) * 0.2))
        self._tax_graph.remove_nodes_from(testing_nodes)
        self.sample_paths()
        # self.data_loader = self._init_data_loader()

    def sample_paths(self):
        self._pos_paths = []
        self._neg_paths = []
        self._margins = []
        paths = [list(reversed(path[1:])) for path in self._tax_graph.all_paths()]
        for path in paths:
            # 这里可以添加根据长度进行sample多少的pos
            self._pos_paths.append(path)
            # 这里添加如何sample neg
            while True:
                neg_node = random.choice(list(self._tax_graph.nodes.keys()))
                if neg_node not in path and neg_node != self._tax_graph.root:
                    break
            self._neg_paths.append([neg_node] + path[1:])
            self._margins.append(self._tax_graph.get_margin(path, neg_node))

    def __len__(self):
        return len(self._pos_paths)

    def __getitem__(self, item):
        pos = self._pos_paths[item]
        neg = self._neg_paths[item]
        margin = self._margins[item]
        pos_ids = self._tokenizer.encode_ids(pos)
        neg_ids = self._tokenizer.encode_ids(neg)
        pos_pool_matrix = self._get_pooling_matrix(pos_ids)
        neg_pool_matrix = self._get_pooling_matrix(neg_ids)
        pos_ids = [pid for ids in pos_ids for pid in ids]
        neg_ids = [nid for ids in neg_ids for nid in ids]
        assert len(pos_ids) <= self._padding_max
        assert len(neg_ids) <= self._padding_max
        pos_attn_masks = [False] * len(pos_ids) + [True] * (self._padding_max - len(pos_ids))
        neg_attn_masks = [False] * len(neg_ids) + [True] * (self._padding_max - len(neg_ids))
        pos_ids.extend([self._padding_id] * (self._padding_max - len(pos_ids)))
        neg_ids.extend([self._padding_id] * (self._padding_max - len(neg_ids)))
        return dict(pos_ids=torch.LongTensor(pos_ids),
                    neg_ids=torch.LongTensor(neg_ids),
                    pos_pool_matrix=pos_pool_matrix,
                    neg_pool_matrix=neg_pool_matrix,
                    pos_attn_masks=torch.BoolTensor(pos_attn_masks),
                    neg_attn_masks=torch.BoolTensor(neg_attn_masks),
                    margin=torch.FloatTensor([margin*0.01]))

    def _get_pooling_matrix(self, ids):
        """
        对于attention后的结果，我们只需要有关于query的部分
        :param ids:
        :return:
        """
        return torch.FloatTensor([1.0 / float(len(ids[0]))] * len(ids[0]) + [0] * (self._padding_max - len(ids[0])))
