"""
Developed by Aindriya Barua in April, 2019
Code for paper https://www.researchgate.net/publication/349190662_Analysis_of_Contextual_and_Non-contextual_Word_Embedding_Models_for_Hindi_NER_with_Web_Application_for_Data_Collection
This project does Named Entity Recognition for Hindi, using Contextual NER Models based on Transformers as dicussed in the paper,
BERT, RoBERTa, ELECTRA, CamemBERT, Distil-BERT, XLM-RoBERTa.

If you use any part of the resources provided in this repo, kindly cite:
Barua, A., Thara, S., Premjith, B. and Soman, K.P., 2020, December. Analysis of Contextual and Non-contextual Word Embedding Models for Hindi NER with Web Application for Data Collection. In International Advanced Computing Conference (pp. 183-202). Springer, Singapore.
"""

def get_dataset(filepath):
    sentence_id, words, labels = get_train_data(filepath)
    dataset=[]
    for i in range(0,len(words)):
        dat=[]
        dat.append(sentence_id[i])
        dat.append(words[i])
        dat.append(labels[i])
        dataset.append(dat)
    return dataset

def get_train_data(filepath):
    texts = []
    labels = []
    sentence_id = []
    read_file = open(filepath, 'r')
    sent_id = 0
    for line in read_file:
        line = line.replace('\n', '')
        if line == 'newline':
            sent_id = sent_id + 1
        else:
            items = line.split('\t')
            sentence_id.append(sent_id)
            texts.append(items[0])
            labels.append(items[1])
    read_file.close()

    return sentence_id, texts, labels


