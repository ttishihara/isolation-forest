import numpy as np
from numpy import random
from scipy import stats
import pandas as pd
from sklearn.metrics import confusion_matrix

# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        height_limit = int(np.ceil(np.log2(sample_size)))
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = [IsolationTree(height_limit) for _ in range(self.n_trees)]


    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.sample_size * self.n_trees < X.shape[0]:
            idx = np.random.randint(X.shape[0], size=self.sample_size * self.n_trees)

            for i, tree in enumerate(self.trees):
                X_ = X[idx[i*self.sample_size: (i+1)*self.sample_size]]
                if improved == False:
                    tree.fit(X_, 0, improved)
                else:
                    tree.fit_improved(X_, 0, improved)
        else:
            for tree in self.trees:
                idx = np.random.randint(X.shape[0], size=self.sample_size)
                X_ = X[idx]
                if improved == False:
                    tree.fit(X_, 0, improved)
                else:
                    tree.fit_improved(X_, 0, improved)
                
        return self


    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        avg_lengths_X = []
        for i, x_i in enumerate(X):
            path_lengths_tree = []
            for t in self.trees:
                path_lengths_tree.append(t.get_path_height(x_i, t.root))
            avg_lengths_X.append(np.mean(path_lengths_tree))

        return np.array(avg_lengths_X)

    
    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        h_x = self.path_length(X)
        if self.sample_size > 2:
            c = 2.0*(np.log(self.sample_size-1.0)+np.euler_gamma) - (2.0*(self.sample_size-1.0)/self.sample_size)
        elif self.sample_size == 2:
            c = 1.0
        else:
            c = 0.0
        
        return 2.0**(-h_x / c)


    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        predictions = np.array([1 if x >= threshold else 0 for x in scores])

        return predictions


    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        if isinstance(X, pd.DataFrame):
            X = X.values
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)


class IsolationTree:
    def __init__(self, height_limit):
        self.n_nodes = 1
        self.height_limit = height_limit

    def fit(self, X:np.ndarray, height = 0, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.height_limit - height == 0 or len(X) <= 1 or (X == X[0]).all(): 
            return TreeNode(None, None, None, None, height, X, 'ex_node')

        split_attr = random.randint(X.shape[1])
        split_value = random.uniform(min(X[:, split_attr]), max(X[:, split_attr]))
        X_l = X[X[:, split_attr] < split_value]
        X_r = X[X[:, split_attr] >= split_value]
        self.n_nodes += 2
        self.root = TreeNode(self.fit(X_l, height+1), 
                             self.fit(X_r, height+1),
                             split_attr, split_value, height, X)

        return self.root

    def fit_improved(self, X:np.ndarray, height = 0, improved=True):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.height_limit - height == 0 or len(X) <= 1 or (X == X[0]).all(): 
            return TreeNode(None, None, None, None, height, X, 'ex_node')

        if X.shape[1] >= 5:
            split_attrs = np.random.randint(X.shape[1], size=5)
        else:
            split_attrs = np.random.randint(X.shape[1], size=X.shape[1])
        
        split_vals = random.uniform(np.min(X[:, split_attrs], axis=0), np.max(X[:, split_attrs], axis=0))

        diffs = []
        split_data = []
        for attr, val in list(zip(split_attrs, split_vals)):
            split = [X[:, attr]<val]
            split_data.append(split)
            diffs.append(X.shape[0] - np.sum(split))

        best_diff_idx = np.argsort(diffs)[0]
        split_attr = split_attrs[best_diff_idx]
        split_val = split_vals[best_diff_idx]
        left_bool = split_data[best_diff_idx]
        
        self.n_nodes += 2
        self.root = TreeNode(self.fit_improved(X[tuple(left_bool)], height+1, improved), 
                             self.fit_improved(X[not tuple(left_bool)], height+1, improved),
                             split_attr, split_val, height, X)

        return self.root


    def get_path_height(self, x_i, node):
        """
        Compute path length for x_i.
        """

        while node.node_type != 'ex_node':
            a = node.split_attr
            if x_i[a] < node.split_val:
                node = node.left
            else:
                node = node.right

        if node.n_objs > 2:
            return node.height + 2.0*(np.log(node.n_objs-1)+np.euler_gamma) - (2.0*(node.n_objs - 1)/node.n_objs)
        elif node.n_objs == 2:
            return node.height + 1
        else:
            return node.height



class TreeNode:
    def __init__(self, left, right, split_attr, split_val, height, data=None, node_type='in_node'):
        self.left = left
        self.right = right
        self.split_attr = split_attr
        self.split_val = split_val
        self.height = height
        self.node_type = node_type

        if node_type == 'ex_node':
            self.n_objs = len(data)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    good_thresholds = []
    for threshold in reversed(np.linspace(0, 1, 101)):
        y_hat = np.zeros((len(scores), 1))
        y_hat[scores >= threshold] = 1
        confusion = confusion_matrix(y, y_hat)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            good_thresholds.append((threshold, FPR))

    good_thresholds = sorted(good_thresholds, key=lambda x: x[1])
    return good_thresholds[0]
