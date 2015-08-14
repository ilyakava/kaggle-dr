import numpy
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
from skimage.io import imread
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import pdb

class UnsupportedPredictedClasses(Exception):
    pass

def assert_valid_prediction(y_pred, K):
    forbidden_klasses = set(y_pred.flatten()) - set(list(range(K)))
    if len(forbidden_klasses):
        raise UnsupportedPredictedClasses(forbidden_klasses)

def save_confusion_matrix_plot(M, labels=None, outpath='plots/conf.png'):
    plt.imshow(M, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title('Confusion matrix')
    plt.colorbar()
    if labels:
        tick_marks = numpy.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label (Accuracy: %.4f)' % (numpy.diag(M).sum() / float(M.sum())))
    print("Saving %s" % outpath)
    plt.savefig(outpath, dpi=600, bbox_inches='tight')

def print_confusion_matrix(M):
    K = M.shape[0]
    print("Confusion Matrix")
    left_margins = [" T\P |", "-----|"]
    header = left_margins[0] + "".join([" %4d |" % klass for klass in range(K)])
    spacer = left_margins[1] + "".join(["------|" for klass in range(K)])
    print header
    print spacer
    for pred_klass, true_klass in enumerate(xrange(K)):
        row = (" %3d |" % true_klass) + "".join([" %4d |" % entry for entry in M[pred_klass]])
        print row
        print spacer

def confusion_matrix_kappa(M,K):
    dx = numpy.ones((K,1)) * numpy.arange(K) # klass rating increasing left to right
    dy = dx.transpose()
    d = (dx - dy)**2 / (K-1)**2
    col_sum = M.sum(axis=0)
    row_sum = M.sum(axis=1)
    E = row_sum.reshape((K,1)).dot(col_sum.reshape((1,K))) / float(M.sum())
    score = 1 - ((d*M).sum()/M.sum()) / ((d*E).sum() / E.sum())
    return score

def QWK(y_true, y_pred, K=5):
    """
    Quadratic mean weighted kappa
    Taken from Matlab code: https://github.com/benhamner/ASAP-AES/blob/master/Evaluation_Metrics/Matlab-Octave/scoreQuadraticWeightedKappa.m
    That was referenced from Admin in Kaggle forums: https://www.kaggle.com/c/asap-aes/forums/t/1289/scoring-metric-verification
    """
    assert_valid_prediction(y_pred, K)
    M = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    score = confusion_matrix_kappa(M,K)
    return [score, M]

def binary_accuracy_precision(true_labels, predicted_labels):
    true_labels = numpy.array(true_labels)
    predicted_labels = numpy.array(predicted_labels)
    accuracy = sum(true_labels == predicted_labels) / float(len(true_labels))
    tp = sum((true_labels + predicted_labels) == 2)
    fp = sum((predicted_labels - true_labels) == 1)
    precision = tp / float(tp + fp)
    return(accuracy, precision)
