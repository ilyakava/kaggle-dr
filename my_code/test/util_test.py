import unittest

from my_code.util import QWK

class Util(unittest.TestCase):
    def testPerfect(self):
        y = [0, 1, 2, 3, 4]
        k = QWK(y, y)
        self.failUnless(k == 1)

    def testAlmostPerfect(self):
        y_true = [4,0,2,3,1,2,2,3]
        y_pred = [4,0,2,3,1,2,3,2]
        k = QWK(y_true, y_pred)
        self.failUnless(round(k,4) == 0.9080)

    def testAwful(self):
        y = [0, 1, 2, 3, 4]
        k = QWK(y, list(reversed(y)))
        self.failUnless(round(k,4) == -1)

def main():
  unittest.main()

if __name__ == '__main__':
  main()
