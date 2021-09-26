"""
Developed by Aindriya Barua in April, 2019
Code for paper https://www.researchgate.net/publication/349190662_Analysis_of_Contextual_and_Non-contextual_Word_Embedding_Models_for_Hindi_NER_with_Web_Application_for_Data_Collection
This project does Named Entity Recognition for Hindi, using either of the two Non-Contextual Word Embedding Models as dicussed in the paper, FastText and Word2Vec, with Classical ML classifiers

If you use any part of the resources provided in this repo, kindly cite:
Barua, A., Thara, S., Premjith, B. and Soman, K.P., 2020, December. Analysis of Contextual and Non-contextual Word Embedding Models for Hindi NER with Web Application for Data Collection. In International Advanced Computing Conference (pp. 183-202). Springer, Singapore.
"""


def get_train_data(path):
    texts = []
    labels = []
    read_file = open(path, 'r', encoding="utf8")
    for line in read_file:
        line = line.replace('\n', '')
        if line == 'newline':
            pass
        else:
            items = line.split('\t')
            texts.append(items[0])
            labels.append(items[1])
    read_file.close()

    return texts, labels
