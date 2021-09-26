# Created by Aindriya Barua at
from pandas import np
from sklearn.preprocessing import label_binarize

from keras.utils.np_utils import to_categorical
from numpy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def get_roc_curve(clf, X_test, y, y_test, label_encoder):
    target_names = ['datenum', 'event', 'location', 'name', 'number', 'occupation', 'organization', 'other', 'things']

    pred = clf.predict(X_test)

    pred_label = label_encoder.inverse_transform(pred)
    print(pred_label)
    y_testcat = to_categorical(y_test, 9)
    y_pred = to_categorical(pred, 9)

    y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(9):
        fpr[i], tpr[i], _ = roc_curve(y_testcat[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_testcat.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    n_classes = 9
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(8)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()

    colors = cycle(['red', 'blue', 'green', 'pink', 'brown', 'aqua', 'black', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.show()