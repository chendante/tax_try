from selenium import webdriver
import codecs
from tqdm import tqdm
import json


def catch_des(word):
    try:
        url = "https://en.wikipedia.org/wiki/" + word.replace(" ", "_")
        driver.get(url)
        test_no_ar = driver.find_elements_by_id("noarticletext")
        if len(test_no_ar) > 0:
            return ""
        content_ele = driver.find_element_by_id("mw-content-text")
        p_list = content_ele.find_elements_by_tag_name("p")
        for p in p_list:
            if p.get_attribute("class") == "" and p.text != "":
                return p.text
    except:
        return ""
    return ""


def search_meanings(word):
    terms = word.split(" ")
    try_list = [word]
    if len(terms) > 1:
        try_list.append(word.replace(" ", "-", 1))
    for w in try_list:
        res = catch_des(w)
        if res != "":
            return res
    return ""


def main():
    with codecs.open(path, 'r', 'utf-8') as fp:
        lines = fp.readlines()
    words = list(set([w for line in lines for w in line.strip().split("\t")[1:]]))
    m_dic = {}
    for w in tqdm(words, total=len(words)):
        meaning = search_meanings(w)
        if meaning == "":
            print(w)
            continue
        m_dic[w] = meaning
    with codecs.open(out_path, "w+") as fp:
        json.dump(m_dic, fp)


def analysis_dic():
    with codecs.open(out_path, "r") as fp:
        dic = json.load(fp)
    count = 0
    for w, des in dic.items():
        if w.lower() in des.lower():
            count += 1
        else:
            print(w, des)
    print(len(dic), count)


def filter_dic():
    with codecs.open(path, 'r', 'utf-8') as fp:
        lines = fp.readlines()
    words = list(set([w for line in lines for w in line.strip().split("\t")[1:]]))
    with codecs.open(out_path, "r") as fp:
        dic = json.load(fp)
    new_dic = {}
    for w, des in dic.items():
        sents = des.lower().split(".")
        res = sents[0]
        for sent in sents:
            if w.lower() in sent:
                res = sent
                break
        new_dic[w] = res
    for w in words:
        if w not in new_dic:
            new_dic[w] = w
    with codecs.open(filter_path, "w+") as fp:
        json.dump(new_dic, fp)


if __name__ == '__main__':
    path = "../data/raw_data/TExEval-2_testdata_1.2/gs_taxo/EN/science_wordnet_en.taxo"
    out_path = "../data/preprocessed/sci_wiki_dic.json"
    filter_path = "../data/preprocessed/f_sci_wiki_dic.json"
    # driver = webdriver.Chrome()
    # main()
    # catch_des("geography")
    filter_dic()
