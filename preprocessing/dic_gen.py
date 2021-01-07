from PyDictionary import PyDictionary
import codecs
import json
from tqdm import tqdm


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
    for w in try_list:
        tag2meaning = dic.meaning(w.replace(" ", "+"), True)
        if tag2meaning is not None:
            res = filter_meanings(tag2meaning)
            if res:
                return res


if __name__ == '__main__':
    path = "../data/raw_data/TExEval-2_testdata_1.2/gs_taxo/EN/science_wordnet_en.taxo"
    out_path = "../data/preprocessed/science_dic.json"
    with codecs.open(path, 'r', 'utf-8') as fp:
        lines = fp.readlines()
    words = list(set([w for line in lines for w in line.strip().split("\t")[1:]]))
    dic = PyDictionary()
    m_dic = {}
    for w in tqdm(words, total=len(words)):
        meanings = search_meanings(w)
        if not meanings:
            print(w)
            continue
        m_dic[w] = meanings
    with codecs.open(out_path, "w+") as fp:
        json.dump(m_dic, fp)
