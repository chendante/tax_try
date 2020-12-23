import os
import argparse
import codecs
from xml.dom.minidom import parseString
from tqdm import tqdm
import json
import re


def convert_taxo2voc(taxo_path, out_path):
    with codecs.open(taxo_path, encoding='utf-8') as f:
        taxo_lines = f.readlines()
    taxo_pairs = [[w for w in line.strip().split("\t")[1:]] for line in taxo_lines]
    voc_list = [w for pair in taxo_pairs for w in pair]
    with codecs.open(out_path, 'w+', encoding='utf-8') as f:
        for voc in voc_list:
            f.write(voc + "\n")


def get_all_path(wiki_dir):
    path_list = os.listdir(wiki_dir)
    all_path = []
    for path in path_list:
        all_path.append(os.path.join(wiki_dir, path))
    return all_path


def read_wiki_file(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as fp:
        content = fp.read()
        content = "<list>" + content + "</list>"
    x = parseString(content)
    return x


def read_voc(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines]


def desc_filter(word: str, full_desc: str):
    # 删除括号内容
    pattern = r"\([^()]*\)"
    full_desc = re.sub(pattern, '', full_desc).lower()
    lines = full_desc.split("\n")
    desc_line = ""
    for line in lines:
        line = line.strip()
        if line == word or line == "":
            continue
        desc_line = line
        break
    sentences = desc_line.split(".")
    res = ""
    for s in sentences:
        res += s
        if len(res.split(" ")) > 5 + len(word.split(" ")):
            break
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dir", type=str, required=True, help="path to wiki preprocessed directory")
    parser.add_argument("--voc_path", type=str, required=True, help="path to vocabulary file")
    parser.add_argument("--out_path", type=str, required=True, help="path to output file")
    args = parser.parse_args()
    doc_dirs = get_all_path(args.wiki_dir)
    voc = read_voc(args.voc_path)
    file_paths = []
    word2des = {}
    for doc_dir in doc_dirs:
        file_paths.extend(get_all_path(doc_dir))
    for file_path in tqdm(file_paths, total=len(file_paths), desc="正在逐个处理文件"):
        xml_data = read_wiki_file(file_path)
        xml_data = xml_data.documentElement
        docs = xml_data.getElementsByTagName('doc')
        if len(docs) == 0:
            print(file_path)
        for doc in docs:
            title = doc.getAttribute("title")
            if title.lower() in voc:
                word2des[title.lower()] = desc_filter(title.lower(), doc.childNodes[0].nodeValue)
    warning_words = [w for w in voc if w not in word2des]
    print("WARNING These Words Not Found:", warning_words)
    with codecs.open(args.out_path, 'w+', encoding='utf-8') as fp:
        json.dump(word2des, fp)


if __name__ == '__main__':
    main()
    # convert_taxo2voc("../data/raw_data/TExEval-2_testdata_1.2/gs_taxo/EN/science_wordnet_en.taxo",
    #                  "../data/preprocessed/science_taxo_words.txt")
    # read_wiki_file("../data/wiki/AA/wiki_00")
