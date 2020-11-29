from gensim.models import word2vec

if __name__ == '__main__':
    sentences = word2vec.LineSentence("../data/preprocessed_corpus/med_connected_lower.txt")
    model = word2vec.Word2Vec(sentences, size=128, window=5, min_count=5, workers=4)
    model.wv.save_word2vec_format("../data/embedding/med_connected_lower_embedding.txt", binary=False)
