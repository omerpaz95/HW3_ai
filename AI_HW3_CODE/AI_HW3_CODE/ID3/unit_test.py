import unittest
import numpy as np
from ID3 import ID3
#from KNN import KNNClassifier
from utils import accuracy, l2_dist


class Test(unittest.TestCase):

    def setUp(self):
        method = self._testMethodName
        print(method + f":{' ' * (25 - len(method))}", end='')

    def test_l2_dist_shape(self):
        mat1 = np.array([[1, 1, 1],
                         [2, 2, 2]])  # (N1, D) = (2, 3)
        mat2 = np.array([[0, 0, 0]])  # (N2, D) = (1, 3)

        l2_dist_result = l2_dist(mat1, mat2)

        self.assertEqual(l2_dist_result.shape, (2, 1))
        print('Success')

    def test_l2_dist_result(self):
        mat1 = np.array([[1, 1, 1],
                         [2, 2, 2]])  # (N1, D) = (2, 3)
        mat2 = np.array([[0, 0, 0]])  # (N2, D) = (1, 3)

        l2_dist_result = l2_dist(mat1, mat2)
        self.assertTrue(np.array_equal(l2_dist_result, np.sqrt([[3], [12]])))
        print('Success')

    def test_accuracy(self):
        y1 = np.array([1, 1, 1])  # (N1, D) = (3,)
        y2 = np.array([1, 0, 0])  # (N2, D) = (3,)

        accuracy_val = accuracy(y1, y2)
        self.assertEqual(accuracy_val, 1 / 3)
        print('Success')

    def test_entropy(self):
        rows = np.array([0,0,0,0,0,0,0,0,0,0])
        labels = np.array(['M','M','M','M','M','M','M','M','M','F'])
        self.assertEqual(len(rows), len(labels))
        entropy = ID3.entropy(rows, labels)
        my_entropy = 0.4689955935892
        self.assertAlmostEqual(entropy, my_entropy, delta=0.001)




if __name__ == '__main__':
    unittest.main()
