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
        self.testing_nodes = random.sample(leaf_nodes, int(len(leaf_nodes) * 0.2))
        self.testing_predecessors = [list(self._tax_graph.predecessors(node)) for node in self.testing_nodes]
        self._tax_graph.remove_nodes_from(self.testing_nodes)
        # path: leaf -> root
        self.paths = [list(reversed(path[1:])) for path in self._tax_graph.all_paths()]
        self.node2path = dict()
        for p in self.paths:
            self.node2path[p[0]] = p
        self.sample_paths()

    def sample_paths(self):
        self._pos_paths = []
        self._neg_paths = []
        self._margins = []
        for path in self.paths:
            # 这里可以添加根据长度进行sample多少的pos
            self._pos_paths.append(path)
            # 这里添加如何sample neg
            while True:
                neg_node = random.choice(list(self._tax_graph.nodes.keys()))
                if neg_node not in path and neg_node != self._tax_graph.root:
                    break
            self._neg_paths.append([neg_node] + path[1:])
            self._margins.append(self._tax_graph.get_margin(path, neg_node))

    def get_eval_data(self):
        path_group = dict()
        for node, predecessors in zip(self.testing_nodes, self.testing_predecessors):
            labels = []
            testing_paths = []
            predecessor = predecessors[0]
            for path in self.paths:
                if path[0] == predecessor:
                    labels.append(True)
                else:
                    labels.append(False)
                testing_paths.append([node] + path)
            path_group[node] = (labels, testing_paths)

    def __len__(self):
        return len(self._pos_paths)

    def __getitem__(self, item):
        pos = self._pos_paths[item]
        neg = self._neg_paths[item]
        margin = self._margins[item]
        pos_ids, pos_pool_matrix, pos_attn_masks = self.encode_path(pos)
        neg_ids, neg_pool_matrix, neg_attn_masks = self.encode_path(neg)
        return dict(pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    pos_pool_matrix=pos_pool_matrix,
                    neg_pool_matrix=neg_pool_matrix,
                    pos_attn_masks=pos_attn_masks,
                    neg_attn_masks=neg_attn_masks,
                    margin=torch.FloatTensor([margin * 0.1]))

    def encode_path(self, path):
        ids = self._tokenizer.encode_ids(path)
        pool_matrix = self._get_pooling_matrix(ids)
        ids = [eid for id_s in ids for eid in id_s]
        assert len(ids) <= self._padding_max
        attn_masks = [False] * len(ids) + [True] * (self._padding_max - len(ids))
        ids.extend([self._padding_id] * (self._padding_max - len(ids)))
        return torch.LongTensor(ids), pool_matrix, torch.BoolTensor(attn_masks)

    def _get_pooling_matrix(self, ids):
        """
        对于attention后的结果，我们只需要有关于query的部分
        :param ids:
        :return:
        """
        return torch.FloatTensor([1.0 / float(len(ids[0]))] * len(ids[0]) + [0] * (self._padding_max - len(ids[0])))
