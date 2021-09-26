"""
Developed by Aindriya Barua in April, 2019
Code for paper https://www.researchgate.net/publication/349190662_Analysis_of_Contextual_and_Non-contextual_Word_Embedding_Models_for_Hindi_NER_with_Web_Application_for_Data_Collection
This project does Named Entity Recognition for Hindi, using either of the two Non-Contextual Word Embedding Models as dicussed in the paper, FastText and Word2Vec, with Classical ML classifiers

If you use any part of the resources provided in this repo, kindly cite:
Barua, A., Thara, S., Premjith, B. and Soman, K.P., 2020, December. Analysis of Contextual and Non-contextual Word Embedding Models for Hindi NER with Web Application for Data Collection. In International Advanced Computing Conference (pp. 183-202). Springer, Singapore.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(y_test, y_pred):
    print("Accuracy score = ", '%.2f' % (accuracy_score(y_test, y_pred) * 100))
    print("Precision = ", '%.4f' % (precision_score(y_test, y_pred, average='macro')))
    print("Recall = ", '%.4f' % (recall_score(y_test, y_pred, average='macro')))
    print("F1-score = ", '%.4f' % (f1_score(y_test, y_pred, average='macro')))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n ", cnf_matrix)
    evaluate_classwise(cnf_matrix)


def evaluate_classwise(cnf_matrix):
    print("Classwise accuracy:\n")
    class_acc = []
    for i in range(len(cnf_matrix)):
        class_acc.append(cnf_matrix[i][i] / cnf_matrix[i].sum() * 100)
    print(class_acc)