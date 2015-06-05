import unittest

from my_code.util import QWK, print_confusion_matrix

class Util(unittest.TestCase):
    def testPerfect(self):
        y = [0, 1, 2, 3, 4]
        k, M = QWK(y, y)
        print_confusion_matrix(M)
        self.failUnless(k == 1)

    def testAlmostPerfect(self):
        y_true = [4,0,2,3,1,2,2,3]
        y_pred = [4,0,2,3,1,2,3,2]
        k, M = QWK(y_true, y_pred)
        self.failUnless(round(k,4) == 0.9080)

    def testAwful(self):
        y = [0, 1, 2, 3, 4]
        k, M = QWK(y, list(reversed(y)))
        self.failUnless(round(k,4) == -1)

def main():
  unittest.main()

if __name__ == '__main__':
  main()
