from selenium import webdriver
import codecs
from tqdm import tqdm
import json
import re


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


def search_meaning(word):
    terms = word.split(" ")
    try_list = [word]
    if len(terms) > 1:
        try_list.append(word.replace(" ", "-", 1))
    for w in try_list:
        res = catch_des(w)
        if res != "":
            return res
    return ""


def get_word_left_info(word):
    if word in forbidden_list:
        return "", ""
    info = search_meaning(word)
    if info != "":
        return word, info
    terms = word.split(" ")
    if len(terms) > 1:
        return get_word_left_info(" ".join(terms[:-1]))
    return "", ""


def get_word_right_info(word):
    if word in forbidden_list:
        return "", ""
    info = search_meaning(word)
    if info != "":
        return word, info
    terms = word.split(" ")
    if len(terms) > 1:
        return get_word_left_info(" ".join(terms[1:]))
    return "", ""


def get_word_info(word: str):
    res = {}
    need_search = word
    while need_search != "":
        left_word, info = get_word_left_info(need_search)
        if left_word != "":
            res[word] = info
            need_search = need_search.replace(left_word + " ", "").replace(left_word, "")
        else:
            need_search = " ".join(need_search.split(" ")[1:])
    return res


def main():
    """
    为taxo中的词，从wikipedia上寻找对应页面，抽取出第一段话。
    """
    with codecs.open(path, 'r', 'utf-8') as fp:
        lines = fp.readlines()
    words = list(set([w for line in lines for w in line.strip().split("\t")[1:]]))
    m_dic = getted_dic
    words = [w for w in words if w not in getted_dic]
    print(len(words), words)
    for w in tqdm(words, total=len(words)):
        meaning = get_word_info(w)
        if meaning == {}:
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
        new_dic[w] = wash_line(res)
    with codecs.open(wordnet_path, "r") as fp:
        wordnet_dic = json.load(fp)
    for w in words:
        if w not in new_dic:
            new_dic[w] = wordnet_dic[w]
            print(w)
    with codecs.open(filter_path, "w+") as fp:
        json.dump(new_dic, fp)


def wash_line(line):
    line = re.sub("\[[0-9]*]", "", line)
    line = re.sub("\(.*\)", "", line)
    return [line]


if __name__ == '__main__':
    forbidden_list = ["a", "the", "is", "'s", "not", "don't", "and"]
    path = "../data/raw_data/TExEval-2_testdata_1.2/gs_taxo/EN/food_wordnet_en.taxo"
    getted_dic = {}
    with codecs.open("../data/preprocessed/food/wiki_dic.json", "r") as fp:
        getted_dic = json.load(fp)
    # getted_dic["marchand de vin"] = "Sauce marchand de vin \"wine-merchant's sauce\" is a similar designation."
    out_path = "../data/preprocessed/food/wiki_full_dic.json"
    # wordnet_path = "../data/preprocessed/science_dic.json"
    filter_path = "../data/preprocessed/food/f_wiki_dic.json"
    driver = webdriver.Chrome()
    main()
    # catch_des("geography")
    # filter_dic()
