import codecs
import collections


class Voc:
    def __init__(self, file_path):
        self.words = collections.defaultdict(list)
        self.file_path = file_path
        with codecs.open(file_path, encoding='utf-8') as f:
            while True:
                word = f.readline()
                if word == "":
                    break
                word = word.strip()
                w_list = word.split(" ")
                if len(w_list) > 1:
                    follow = w_list[1:]
                    self.words[w_list[0]].append(follow)

    def match_word(self, head):
        return self.words.get(head)


def tag(corpus_path, out_path, voc_path):
    """
    转小写 + 多节词连接
    :param corpus_path:
    :param out_path:
    :param voc_path:
    :return:
    """
    voc = Voc(voc_path)
    out_writer = codecs.open(out_path, 'x', 'utf-8')
    with codecs.open(corpus_path, 'r', encoding='utf-8') as f:
        count = 0
        tagged_count = 0
        while True:
            line = f.readline()
            count += 1
            print(count, tagged_count)
            if line == "":
                break
            words = line.lower().split(" ")
            skip = 0
            tagged_list = []
            for i, w in enumerate(words):
                if skip > 0:
                    skip -= 1
                    tagged_list[-1] += "_" + w
                    continue
                tagged_list.append(w)
                follows = voc.match_word(w)
                if follows is not None:
                    # 这里需要反转是因为这样可以更长的单词先进行匹配
                    # 比如 red big apple 和 red big 就可以匹配到 red big apple 而非 red big
                    for follow in reversed(follows):
                        if len(follow) + i + 1 < len(words):
                            if follow == words[i + 1: i + 1 + len(follow)]:
                                skip = len(follow)
                                tagged_count += 1
                                print(w, follow)
                                break
            tagged_line = " ".join(tagged_list)
            out_writer.write(tagged_line)
        out_writer.flush()
        out_writer.close()
    return tagged_count


if __name__ == '__main__':
    print(tag("../data/raw_data/2A_med_pubmed_tokenized.txt",
              "../data/preprocessed_corpus/med_connected_lower.txt",
              "../data/raw_data/SemEval2018-Task9/vocabulary/2A.medical.vocabulary.txt"))
