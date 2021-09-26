# Created by Aindriya Barua at
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