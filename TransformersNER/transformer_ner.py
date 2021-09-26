"""
Developed by Aindriya Barua in April, 2019
Code for paper https://www.researchgate.net/publication/349190662_Analysis_of_Contextual_and_Non-contextual_Word_Embedding_Models_for_Hindi_NER_with_Web_Application_for_Data_Collection
This project does Named Entity Recognition for Hindi, using Contextual NER Models based on Transformers as dicussed in the paper,
BERT, RoBERTa, ELECTRA, CamemBERT, Distil-BERT, XLM-RoBERTa.

If you use any part of the resources provided in this repo, kindly cite:
Barua, A., Thara, S., Premjith, B. and Soman, K.P., 2020, December. Analysis of Contextual and Non-contextual Word Embedding Models for Hindi NER with Web Application for Data Collection. In International Advanced Computing Conference (pp. 183-202). Springer, Singapore.
"""

import sys

import dataset_maker
import model_evaluator

from simpletransformers.ner import NERModel
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split


import pickle

def split_data(dataset):
    train_data, eval_data = train_test_split(dataset, test_size=0.20, random_state=42)
    train_df = pd.DataFrame(train_data, columns=['sentence_id', 'words', 'labels'])
    eval_df = pd.DataFrame(eval_data, columns=['sentence_id', 'words', 'labels'])
    return train_df, eval_df


def train_and_predict(embedding, train_df):
    embedding = embedding.lower()
    if ('bert' == embedding):
        model_code = 'bert'
        variant = 'bert-base-uncased'
    else:
        if ('roberta' == embedding):
            model_code = 'bert'
            variant = 'roberta-base'
        else:
            if ('camembert' == embedding):
                model_code = 'camembert'
                variant = 'camembert-base'
            else:
                if ('electra' == embedding):
                    model_code = 'camembert'
                    variant = 'electra-base'
                else:
                    if ('distilbert' == embedding):
                        model_code = 'distilbert'
                        variant = 'distilbert-base-uncased'
                    else:
                        if ('xlm-roberta' == embedding):
                            model_code = 'xlm-roberta'
                            variant = 'xlm-roberta-base'

    model = NERModel(model_code, variant,
                     labels=["datenum", "event", "location", "name", "number", "occupation", "organization", "other",
                             "things"],
                     args={'reprocess_input_data': True, 'save_eval_checkpoints': True, 'num_train_epochs': 4},
                     use_cuda=False)
    model.train_model(train_df)
    pickle.dump(model, open(embedding + '_model.pkl', 'wb'))

    return model


if __name__ == '__main__':
    np.random.seed(1)

    embedding = sys.argv[1]
    dataset_file = sys.argv[2]

    dataset = dataset_maker.get_dataset(dataset_file)

    train_df, eval_df = split_data(dataset)
    model = train_and_predict(embedding, train_df)
    model_evaluator.evaluate(model, eval_df)
