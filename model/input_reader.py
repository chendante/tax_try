import codecs
import networkx as nx
import matplotlib.pyplot as plt
from model.sampler import TaxStruct


class InputReader:
    def __init__(self, taxo_path):
        with codecs.open(taxo_path, encoding='utf-8') as f:
            taxo_lines = f.readlines()
        self.taxo_pairs = [[w.replace(" ", "_") for w in line.strip().split("\t")[1:]] for line in taxo_lines]
        self.taxo_pairs = [(p[1], p[0]) for p in self.taxo_pairs]


if __name__ == '__main__':
    print([1]+[1, 2])
    # I = InputReader("../data/raw_data/TExEval-2_testdata_1.2/gs_taxo/EN/science_wordnet_en.taxo")
    # t = TaxStruct(I.taxo_pairs)
    # # fig, ax = plt.subplots()
    # # nx.draw(t, ax=ax)
    # # plt.show()
    # t.check_useless_edge()
    # print("----")
    # t.check_useless_edge()
    # print(len(t.nodes), len(t.all_leaf_nodes()))
    # w_lengths = nx.shortest_path_length(t, t.root)
    # for node in t.nodes:
    #     if len(t[node]) == 0:
    #         print(node, w_lengths[node])
