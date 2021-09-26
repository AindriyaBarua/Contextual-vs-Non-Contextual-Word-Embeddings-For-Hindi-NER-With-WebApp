# Created by Aindriya Barua at
import sys


def embed_words(words, word_embedding):
    train_data = []
    if "fasttext" == word_embedding:
        import fasttext
        model = fasttext.train_unsupervised("hindi_ner_4.txt", model='skipgram', lr=0.05, dim=100, ws=5, epoch=5)
        model.save_model("fasttext_model_file.bin")

        for i in range(len(words[0])):
            train_data.append(model[words[0][i]])

    if "word2vec" == word_embedding:
        from gensim.models import Word2Vec
        model = Word2Vec(words, vector_size=100, window=1, min_count=1, workers=4)
        model.train(words, total_examples=model.corpus_count, epochs=model.epochs)

        for i in range(len(words[0])):
            train_data.append(model.wv[words[0][i]])
    X = train_data
    return X


def encode_labels(labels):
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.transform(labels)
    return y, label_encoder
