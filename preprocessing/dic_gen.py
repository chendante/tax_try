from PyDictionary import PyDictionary
import codecs
import json
from tqdm import tqdm
from collections import defaultdict


def filter_meanings(t):
    res = []
    for k, v in t.items():
        if k == "Noun":
            return v
        else:
            res.extend(v)
    return res


def search_meanings(word: str):
    terms = word.split(" ")
    try_list = [word]
    if len(terms) > 1:
        try_list.append(word.replace(" ", "-", 1))
    if len(terms) > 2:
        try_list.append(word.replace(" ", "-"))
    for w in try_list:
        tag2meaning = PyDictionary.meaning(w.replace(" ", "+"), True)
        if tag2meaning is not None:
            res = filter_meanings(tag2meaning)
            if res:
                return res


def main():
    with codecs.open(path, 'r', 'utf-8') as fp:
        lines = fp.readlines()
    words = list(set([w for line in lines for w in line.strip().split("\t")[1:]]))
    m_dic = {}
    for w in tqdm(words, total=len(words)):
        meanings = search_meanings(w)
        if not meanings:
            print(w)
            continue
        m_dic[w] = meanings
    with codecs.open(out_path, "w+") as fp:
        json.dump(m_dic, fp)


def check_app():
    with codecs.open(path, 'r', 'utf-8') as fp:
        lines = fp.readlines()
    w2pre = defaultdict(list)
    pre_set = set()
    for line in lines:
        w_p = line.strip().split("\t")[1:]
        w2pre[w_p[0]].append(w_p[1])
        pre_set.add(w_p[1])
    with codecs.open(out_path, "r") as fp:
        w2des = json.load(fp)
    count = 0
    pro_num = 0
    for w, des in w2des.items():
        if w not in pre_set:
            pro_num += 1
            for pre in w2pre[w]:
                if pre in des[0].lower():
                    print(w, pre, des[0])
                    count += 1
    print(pro_num, count)


def append_words(words):
    with codecs.open(out_path, "r") as fp:
        origin_dic = json.load(fp)
    for w in words:
        meanings = search_meanings(w)
        origin_dic[w] = meanings
    with codecs.open(out_path, "w+") as fp:
        json.dump(origin_dic, fp)


if __name__ == '__main__':
    path = "../data/raw_data/TExEval-2_testdata_1.2/gs_taxo/EN/food_wordnet_en.taxo"
    out_path = "../data/preprocessed/food/wordnet_food_dic.json"
    # main()
    app_words = ["cock-a-leekie", "pork-and-veal goulash", "coquilles saint-jacques", "pot-au-feu",
                 "bacon-lettuce-tomato sandwich", "half-and-half dressing", "half-and-half"]
    append_words(app_words)
    # check_app()
    # print(PyDictionary.synonym("science"))
