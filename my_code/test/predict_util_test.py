import unittest
import numpy

from my_code.predict_util import QWK, print_confusion_matrix

class Util(unittest.TestCase):
    def testPerfect(self):
        y = numpy.array([0, 1, 2, 3, 4])
        k, M = QWK(y, y)
        # print_confusion_matrix(M)
        self.failUnless(k == 1)

    def testAlmostPerfect(self):
        y_true = numpy.array([4,0,2,3,1,2,2,3])
        y_pred = numpy.array([4,0,2,3,1,2,3,2])
        k, M = QWK(y_true, y_pred)
        self.failUnless(round(k,4) == 0.9080)

    def testAwful(self):
        y = numpy.array([0, 1, 2, 3, 4])
        yrev = numpy.array(list(reversed(y)))
        k, M = QWK(y, yrev)
        self.failUnless(round(k,4) == -1)

def main():
  unittest.main()

if __name__ == '__main__':
  main()
