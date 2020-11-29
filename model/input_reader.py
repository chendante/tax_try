import torch
import codecs
import networkx as nx


class InputReader:
    def __init__(self, train_d_path, train_g_path):
        with codecs.open(train_g_path, encoding='utf-8') as f:
            g_lines = f.readlines()
        self.g_words = [[w.replace(" ", "_") for w in line.strip("\n").split("\t")] for line in g_lines]
        self.tax_graph = build_taxonomy(self.g_words, threshold=1.0, num_bar=3, add_num_bar=10)
        with codecs.open(train_d_path, encoding='utf-8') as f:
            d_lines = f.readlines()
        self.d_words = [line.strip("\n").split("\t")[0].replace(" ", "_") for line in d_lines]
