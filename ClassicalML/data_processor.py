"""
Developed by Aindriya Barua in April, 2019
Code for paper https://www.researchgate.net/publication/349190662_Analysis_of_Contextual_and_Non-contextual_Word_Embedding_Models_for_Hindi_NER_with_Web_Application_for_Data_Collection
This project does Named Entity Recognition for Hindi, using either of the two Non-Contextual Word Embedding Models as dicussed in the paper, FastText and Word2Vec, with Classical ML classifiers

If you use any part of the resources provided in this repo, kindly cite:
Barua, A., Thara, S., Premjith, B. and Soman, K.P., 2020, December. Analysis of Contextual and Non-contextual Word Embedding Models for Hindi NER with Web Application for Data Collection. In International Advanced Computing Conference (pp. 183-202). Springer, Singapore.
"""


import pickle


def embed_words(words, word_embedding):
    print("Word Embedding: ", word_embedding)

    train_data = []
    if "fasttext" == word_embedding:
        import fasttext
        model = fasttext.train_unsupervised("hindi_ner_4.txt", model='skipgram', lr=0.05, dim=100, ws=5, epoch=5)
        pickle.dump(model, open('fasttext_model.pkl', 'wb'))

        for i in range(len(words[0])):
            train_data.append(model[words[0][i]])

    if "word2vec" == word_embedding:
        from gensim.models import Word2Vec
        model = Word2Vec(words, vector_size=100, window=1, min_count=1, workers=4)
        model.train(words, total_examples=model.corpus_count, epochs=model.epochs)
        pickle.dump(model, open('word2vec_model.pkl', 'wb'))

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
