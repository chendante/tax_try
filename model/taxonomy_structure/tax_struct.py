from collections import defaultdict
from typing import List
import networkx as nx


class TaxStruct:
    root = "<root>"
    to_hyper = "to_hyper"

    def __init__(self, gold_words: List[List[str]], threshold=1.0, num_bar=3, add_num_bar=10):
        self._add_num_bar = add_num_bar
        self._num_bar = num_bar
        self._threshold = threshold
        self._digraph = nx.DiGraph()
        voc_map = TaxStruct._get_voc_map(gold_words)
        tax_rs, tax_entities = TaxStruct._get_taxonomy_relation(voc_map, threshold, num_bar, add_num_bar)
        same_dict = TaxStruct._get_same(tax_rs)
        edges = TaxStruct._get_edges(tax_rs, same_dict, tax_entities)
        self.add_edges_from(edges)

    def add_edges_from(self, edges):
        self._digraph.add_edges_from(edges, to_hyper=True)
        self._digraph.add_edges_from([(edge[1], edge[0]) for edge in edges], to_hyper=True)

    def get_bros(self, node):
        bros = []
        hypers = [hyper for hyper, tag in self._digraph[node].items() if tag[self.to_hyper]]
        for hyper in hypers:
            for bro, tag in self._digraph[hyper].items():
                if not tag[self.to_hyper]:
                    bros.append(bro)
        return bros

    def has_node(self, node):
        return self._digraph.has_node(node)

    @staticmethod
    def _get_voc_map(gold_words: List[List[str]]):
        voc_map = defaultdict(lambda: defaultdict(lambda: 0))
        for words in gold_words:
            for w in words:
                for w_w in words:
                    voc_map[w][w_w] += 1
        return voc_map

    @staticmethod
    def _get_taxonomy_relation(voc_map, threshold=1.0, num_bar=3, add_num_bar=10):
        """
        :param add_num_bar:
        :param voc_map:
        :param threshold:
        :param num_bar:
        :return: taxonomy_relation: Dict[hyper: List[hypo]]
        """
        taxonomy_relation = defaultdict(list)
        taxonomy_entities = set()

        for word, show_times in voc_map.items():
            all_num = show_times[word]  # 该词自己出现的次数
            if all_num < num_bar:
                continue
            for hyper_word, num in show_times.items():  # num 为sub_v 与它一同出现的次数
                if hyper_word != word and (num / all_num) >= threshold:
                    taxonomy_relation[hyper_word].append(word)
                    taxonomy_entities.add(word)
                    taxonomy_entities.add(hyper_word)
        for word, show_times in voc_map.items():
            if word not in taxonomy_entities and show_times[word] >= add_num_bar:
                taxonomy_relation[word] = []
                taxonomy_entities.add(word)
        return taxonomy_relation, taxonomy_entities

    @staticmethod
    def _get_same(tax_relation):
        same_dict = defaultdict(list)
        for t_w, t_rs in tax_relation.items():
            for t_r in t_rs:
                if t_r not in tax_relation:
                    continue
                rr = tax_relation[t_r]
                if t_w in rr:
                    # print(t_w, " ~~~ ", t_r)
                    same_dict[t_w].append(t_r)
        return same_dict

    @classmethod
    def _get_edges(cls, tax_relation, same_dict, tax_entities):
        edges = []  # hyper to hypo
        for tax_word, relations in tax_relation.items():
            for r in relations:
                if tax_word in same_dict and r in same_dict[tax_word]:
                    continue
                edges.append((tax_word, r))
        children = [e[1] for e in edges]
        for entity in tax_entities:
            if entity not in children:
                edges.append((cls.root, entity))
        return edges

    # @staticmethod
    # def build_taxonomy(gold_words: List[List[str]], threshold=1.0, num_bar=3, add_num_bar=10):
    #     voc_map = TaxStruct.get_voc_map(gold_words)
    #     tax_rs, tax_entities = TaxStruct.get_taxonomy_relation(voc_map, threshold, num_bar, add_num_bar)
    #     same_dict = TaxStruct.get_same(tax_rs)
    #     edges = TaxStruct.get_edges(tax_rs, same_dict, tax_entities)
    #     tax_graph = nx.DiGraph()
    #     tax_graph.add_edges_from(edges)
    #     tax_graph.add_edges_from([(edge[1], edge[0], {"to_hyper": True}) for edge in edges])
    #     return tax_graph
