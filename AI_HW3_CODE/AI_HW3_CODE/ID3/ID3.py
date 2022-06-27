from cProfile import label
import math

from more_itertools import all_equal

from DecisonTree import Leaf, Question, DecisionNode, class_counts, unique_vals
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        #OMER: Done.
        
        counts = class_counts(rows, labels)

        impurity = 0.0
        keys = counts.keys()
        ammount_of_rows = len(rows)
        for key in keys:
            if counts[key] == 0:
                continue
            probability = counts[key] / ammount_of_rows
            impurity += probability * (np.log2(probability))
        return -impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # OMER: Done.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        
        entropy_left = self.entropy(left, left_labels)
        entropy_right = self.entropy(right, right_labels)
        left_size = len(left_labels)
        right_size = len(right_labels)
        info_gain_value = current_uncertainty - (entropy_left * (left_size / (left_size + right_size)) + entropy_right * (right_size / (left_size + right_size)))
        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.
        
        # OMER: I think I'm done.
        gain = 0
        true_rows, true_labels, false_rows, false_labels = [], [], [], []
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'
        for (row, label) in zip(rows, labels):
            if question.match(row):
                true_rows.append(row)
                true_labels.append(label)
            else:
                false_rows.append(row)
                false_labels.append(label)
        gain = self.info_gain(false_rows, false_labels, true_rows, true_labels, current_uncertainty)
        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)

        # Omer: I think I'm done.

        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # First, let's create a list of questions!
        questions = []
        columns =  []
        ammount_of_columns = len(rows[0])
        for i in range(ammount_of_columns):
            columns.append(unique_vals(rows, i))
        
        # Generate questions using different discretizations:
        for i, col in enumerate(columns):
            for val in col:
                question = Question(col, i, val)
                questions.append(question)

        #Then, find the best partition by the best question, by finding the max information gain:
        for question in questions:
            gain, true_rows, true_labels, false_rows, false_labels = self.partition(rows, labels, question, current_uncertainty)
            if gain >= best_gain:
                if gain == best_gain and question.column_idx < best_question.column_idx:
                    continue
                best_question = question
                best_true_rows, best_true_labels = true_rows, true_labels
                best_false_rows, best_false_labels = false_rows, false_labels
                best_gain = gain

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def all_equal(iterator):
        iterator = iter(iterator)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first == x for x in iterator)

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        
        # OMER: I think I have to add leaves...
        best_question = None
        true_branch, false_branch = None, None
        if all_equal(labels):
            return Leaf(rows, labels)
        gain, best_question, true_rows, true_labels, false_rows, false_labels = self.find_best_split(rows, labels)

        true_branch = self.build_tree(true_rows, true_labels)
        false_branch = self.build_tree(false_rows, false_labels)
        return DecisionNode(best_question, true_branch, false_branch)


    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root
        
        # OMER: I think I'm done.

        self.tree_root = self.build_tree(x_train, y_train)
        # print("Training done! Tree_root is of type: ", type(tree_root))

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        # OMER: I think I'm done.
        if node is None:
            node = self.tree_root
        prediction = None

        if type(node) is DecisionNode:
            question = node.question
            if question.match(row):
                return self.predict_sample(row, node.true_branch)
            else:
                return self.predict_sample(row, node.false_branch)
        else:
            preds = node.predictions
            prediction = list(preds.keys())[0]
            # if preds[self.label_names[0]] > preds[self.label_names[1]]:
            #     prediction = self.label_names[0]
            # else:
            #     prediction = self.label_names[1]
        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = np.zeros(shape=(len(rows), ), dtype=type(self.label_names[0]))
        for i, row in enumerate(rows):
            prediction = self.predict_sample(row)
            print("Prediction: ", prediction)
            y_pred[i] = prediction
        return y_pred
