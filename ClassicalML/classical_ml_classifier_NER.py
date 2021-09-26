"""
Developed by Aindriya Barua in April, 2019


This is the starting point of the project which has to be run. It calls the file reader, scraper, output writer
"""
import sys

import read_input_file
import roc_curve_maker
import data_processor
import model_evaluator

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test


def get_class_weights(y):
    unique, counts = np.unique(y, return_counts=True)
    class_weights = dict(zip(unique, np.round(sum(counts) / counts)))
    return class_weights


def fit_classifier(X_train, y_train):

    clfs = {
        #'mnb': MultinomialNB(),
        'gnb': GaussianNB(),
        #'svm1': SVC(kernel='linear'),
        #'svm2': SVC(kernel='rbf'),
        #'svm3': SVC(kernel='sigmoid'),
        'mlp1': MLPClassifier(),
        #'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
        'ada': AdaBoostClassifier(),
        'dtc': DecisionTreeClassifier(),
        'rfc': RandomForestClassifier(),
        #'gbc': GradientBoostingClassifier(),
        #'lr': LogisticRegression()
    }

    for clf_name in clfs:
        print(clf_name)
        clf = clfs[clf_name]
        clf.fit(X_train, y_train)# class_weight=class_weights)
    return clf


def predict_tags(X_test, clf):
    y_pred = clf.predict(X_test)
    return y_pred


if __name__ == '__main__':
    np.random.seed(1)

    word_embedding = sys.argv[1]
    dataset_file = sys.argv[2]
    words, labels = read_input_file.get_train_data(dataset_file)
    words = [words]
    X = data_processor.embed_words(words, word_embedding)
    y, label_encoder = data_processor.encode_labels(labels)
    X_train, X_test, y_train, y_test = split_data(X, y)
    #class_weights = get_class_weights(y)
    clf = fit_classifier(X_train, y_train)#, class_weights)
    y_pred = predict_tags(X_test, clf)
    model_evaluator.evaluate(y_test, y_pred)
    roc_curve_maker.get_roc_curve(clf, X_test, y, y_test, label_encoder)
