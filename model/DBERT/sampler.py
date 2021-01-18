import torch
from typing import *
import model
from torch.utils.data import DataLoader, Dataset
import networkx as nx
import random
import transformers
import json


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

    def all_paths(self, less_len):
        """
        :param less_len: 最小长度
        :return:
        """
        paths = []
        for node in self.nodes:
            path = nx.shortest_path(self, source=self.root, target=node)
            if len(path) > less_len:
                paths.append(path)
        return paths

    def all_leaf_paths(self, less_len):
        """
        所有到当前叶节点的path
        :param less_len: 最小长度
        :return:
        """
        paths = []
        for node in self.all_leaf_nodes():
            path = nx.shortest_path(self, source=self.root, target=node)
            if len(path) > less_len:
                paths.append(path)
        return paths

    @staticmethod
    def get_margin(path_a, path_b):
        com = len(set(path_a).intersection(set(path_b)))
        return abs(len(path_a) - com) + abs(len(path_b) - com)

    def get_node2full_path(self):
        node2full_path = {}
        for node in self.nodes:
            paths = nx.all_simple_paths(self, source=self.root, target=node)
            all_nodes = set([node for path in paths for node in path[1:]])
            node2full_path[node] = all_nodes
        return node2full_path


class Sampler(Dataset):
    def __init__(self, edges, tokenizer: transformers.BertTokenizer, dic_path, padding_max=256, margin_beta=0.05,
                 r_seed=0):
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
        self._margin_beta = margin_beta
        with open(dic_path, 'r', encoding='utf-8') as fp:
            self._word2des = json.load(fp)  # word: ["des0",]
        # tax graph init
        self._tax_graph = TaxStruct(edges)
        leaf_nodes = self._tax_graph.all_leaf_nodes()
        self._seed = r_seed
        random.seed(self._seed)
        self.testing_nodes = random.sample(leaf_nodes, int(len(leaf_nodes) * 0.2))
        self.testing_predecessors = [list(self._tax_graph.predecessors(node)) for node in self.testing_nodes]
        self._tax_graph.remove_nodes_from(self.testing_nodes)
        # path: leaf -> root paths information init
        self.paths = [list(reversed(path[1:])) for path in self._tax_graph.all_paths(2)]
        self.leaf_paths = [list(reversed(path[1:])) for path in self._tax_graph.all_leaf_paths(2)]
        # 用于eval部分
        self.att_paths = [list(reversed(path[1:])) for path in self._tax_graph.all_paths(1)]  # 不带 root
        self.node2full_path = self._tax_graph.get_node2full_path()
        self.node2path = dict()  # 不带 root
        self.index2node = []
        for p in self.att_paths:
            self.node2path[p[0]] = p
            self.index2node.append(p[0])
        self.sample_paths()
        self._eval_path_group = None

    def sample_paths(self):
        self._margins = []
        self._pos_paths = []
        self._neg_paths = []
        for path in self.leaf_paths:
            self._pos_paths.append(path)
            neg_path, margin = self.get_neg_path_in_path(path)
            self._neg_paths.append(neg_path)
            self._margins.append(margin)

    def get_neg_path_in_node(self, pos_path):
        """
        将path的第一个node修改
        """
        while True:
            neg_node = random.choice(list(self._tax_graph.nodes.keys()))
            if neg_node not in pos_path and neg_node != self._tax_graph.root:
                break
        return [neg_node] + pos_path[1:], self._tax_graph.get_margin(pos_path[1:], self.node2path[neg_node][1:])

    def get_neg_path_in_path(self, pos_path):
        """
        保留path的第一个node，修改其后的path
        """
        while True:
            neg_node = random.choice(list(self._tax_graph.nodes.keys()))
            if neg_node != pos_path[0] and neg_node != self._tax_graph.root:
                break
        neg_path = self.node2path[neg_node]
        return pos_path[0:1] + neg_path, self._tax_graph.get_margin(pos_path[1:], neg_path)

    def get_wu_p(self, index_a, index_b):
        if index_a == index_b:
            return 1.0
        node_a = self.index2node[index_a]
        node_b = self.index2node[index_b]
        full_path_a = self.node2full_path[node_a]
        full_path_b = self.node2full_path[node_b]
        com = full_path_a.intersection(full_path_b)
        lca_dep = 0
        for node in com:
            if len(self.node2path[node]) > lca_dep:
                lca_dep = len(self.node2path[node])
        dep_a = len(self.node2path[node_a])
        dep_b = len(self.node2path[node_b])
        res = 2.0 * float(lca_dep) / float(dep_a + dep_b)
        # assert res <= 1
        return res

    def get_eval_data(self):
        if self._eval_path_group is not None:
            return self._eval_path_group
        path_group = dict()
        for node, predecessors in zip(self.testing_nodes, self.testing_predecessors):
            label = -1
            ids_list = []
            token_type_ids_list = []
            attn_masks = []
            predecessor = predecessors[0]
            for i, path in enumerate(self.att_paths):
                if path[0] == predecessor:
                    label = i
                ids, token_type_ids, attn_mask = self.encode_path([node] + path)
                ids_list.append(ids)
                token_type_ids_list.append(token_type_ids)
                attn_masks.append(attn_mask)
            assert label >= 0
            path_group[node] = dict(ids=torch.stack(ids_list, dim=0),
                                    token_type_ids=torch.stack(token_type_ids_list, dim=0),
                                    attn_masks=torch.stack(attn_masks, dim=0), label=label)
        self._eval_path_group = path_group
        return path_group

    def __len__(self):
        return len(self._pos_paths)

    def __getitem__(self, item):
        pos_path = self._pos_paths[item]
        neg_path = self._neg_paths[item]
        margin = self._margins[item]
        pos_ids, pos_type_ids, pos_attn_masks = self.encode_path(pos_path)
        neg_ids, neg_type_ids, neg_attn_masks = self.encode_path(neg_path)
        return dict(pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    pos_type_ids=pos_type_ids,
                    neg_type_ids=neg_type_ids,
                    pos_attn_masks=pos_attn_masks,
                    neg_attn_masks=neg_attn_masks,
                    margin=torch.FloatTensor([margin * self._margin_beta]))

    def encode_path(self, path):
        des_sent = self._word2des[path[0]][0]
        def_sent = " ".join(path)
        encode = self._tokenizer.encode_plus(des_sent, def_sent, add_special_tokens=True,
                                             # return_tensors='pt'
                                             )
        input_len = len(encode["input_ids"])
        assert input_len <= self._padding_max
        encode["input_ids"] = encode["input_ids"] + [self._tokenizer.pad_token_id] * (self._padding_max - input_len)
        encode["token_type_ids"] = encode["token_type_ids"] + [0] * (self._padding_max - input_len)
        encode["attention_mask"] = encode["attention_mask"] + [0] * (self._padding_max - input_len)
        return torch.LongTensor(encode["input_ids"]), torch.LongTensor(encode["token_type_ids"]), torch.LongTensor(
            encode["attention_mask"])
