import numpy
from sklearn.metrics import confusion_matrix

import pdb

K = 5 # num classes

def print_confusion_matrix(M):
    print("Confusion Matrix")
    print("T\P|    0 |    1 |    2 |    3 |    4 |")
    print("---|------|------|------|------|------|")
    for i, y in enumerate(xrange(K)):
        print(" %d | %4d | %4d | %4d | %4d | %4d |" % ((y, ) + tuple(M[i])))
        print("---|------|------|------|------|------|")

def QWK(y_true, y_pred):
    """
    Quadratic mean weighted kappa
    Taken from Matlab code: https://github.com/benhamner/ASAP-AES/blob/master/Evaluation_Metrics/Matlab-Octave/scoreQuadraticWeightedKappa.m
    That was referenced from Admin in Kaggle forums: https://www.kaggle.com/c/asap-aes/forums/t/1289/scoring-metric-verification
    """
    M = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    dx = numpy.ones((K,1)) * numpy.arange(K) # klass rating increasing left to right
    dy = dx.transpose()
    d = (dx - dy)**2 / (K-1)**2
    col_sum = M.sum(axis=0)
    row_sum = M.sum(axis=1)
    E = row_sum.reshape((K,1)).dot(col_sum.reshape((1,K))) / float(M.sum())
    Ef = E.flatten()
    Mf = M.flatten()
    df = d.flatten()
    score = 1 - (sum(df * Mf)/sum(Mf)) / (sum(df * Ef) / sum(Ef))
    return [score, M]
