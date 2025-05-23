'''
Decision Tree
To classify (or regress) data points by learning a hierarchical set of simple "if-then-else" decision rules based on the features. The model partitions the feature space into a set of rectangular regions.

Estimation: The dataset's pattern is learned by recursively splitting the data into subsets based on the feature value that best separates the target variable. This process creates a tree-like structure.
Each path from the root to a leaf node represents a sequence of decisions leading to a specific prediction.

Key Concepts / Structure:

    Nodes:

        Root Node: The topmost node representing the entire dataset before any splits.

        Internal Node (Decision Node): Represents a test on a specific feature (e.g., "Is feature X_i <= threshold?"). It has branches leading to child nodes.

        Leaf Node (Terminal Node): Represents the final outcome or prediction. For classification, this is typically a class label. For regression, it's a continuous value.

    Branches: Connect nodes, representing the outcome of a test at an internal node.

    Splitting: The process of dividing a node into two or more child nodes based on a chosen feature and threshold.

    Depth: The length of the longest path from the root node to a leaf node.

To determine the "best" split at each node, a criterion is used to measure how well a feature and threshold separate the data into "purer" subsets (i.e., subsets that are more homogeneous with respect to the target variable). Common criteria include:

    Information Gain (based on Entropy): Measures the reduction in uncertainty (entropy) after a dataset is split on an attribute. Used in your provided code.

        Entropy(S) = - Σ p_i * log2(p_i) where p_i is the proportion of samples belonging to class i.

        InformationGain(S, A) = Entropy(S) - Σ (|S_v| / |S|) * Entropy(S_v) for a split A creating subsets S_v.

    Gini Impurity: Measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the subset.

        Gini(S) = 1 - Σ p_i^2

"Optimization" Technique: Greedy Recursive Partitioning
The tree is built in a top-down, greedy fashion:

    Start with all data at the root node.

    At each node, find the best feature and threshold to split the data. "Best" is determined by the chosen splitting criterion (e.g., maximizing Information Gain).

    This process is applied recursively to the resulting child nodes.
    It's "greedy" because it makes the locally optimal decision at each step without looking ahead to see if a different split might lead to a globally better tree.

Key Hyperparameters (Controlling tree structure and preventing overfitting):

    max_depth: The maximum allowed depth of the tree. Limits how many consecutive questions can be asked. (e.g., max_depth=100 in your code)

    min_samples_split: The minimum number of samples a node must have to be considered for splitting. (e.g., min_samples_split=2 in your code)

    min_samples_leaf: The minimum number of samples required to be at a leaf node. (Not explicitly in your __init__ but min_samples_split often serves a similar purpose; a split must leave at least this many samples in each child node in some implementations).

    n_features (as used in your code): The number of features to consider when looking for the best split at each node. If less than the total number of features, a random subset is chosen, which can sometimes improve robustness or speed. (e.g., n_features=None in your code, defaulting to all features unless specified).

    criterion: The function to measure the quality of a split (e.g., "entropy" for Information Gain, or "gini"). (Implicitly "entropy" in your code via _information_gain).

Steps:
I. Training Phase (Building the Tree - fit and _grow_tree methods):

The goal is to determine the optimal sequence of feature tests (internal nodes) and their corresponding thresholds, leading to predictive leaf nodes.

    Initialization:

        Define hyperparameters: max_depth, min_samples_split, n_features.

        The root node is initially None. The process starts with the entire training dataset (X, y).

    Recursive Tree Growth (_grow_tree(X, y, depth) function):

    a. Check Stopping Criteria:
    - If current depth >= max_depth.
    - If all samples in the current node belong to the same class (n_labels == 1).
    - If the number of samples in the current node (n_samples) < min_samples_split.
    - If any of these conditions are met:
    - Create a Leaf Node.
    - The value of this leaf node is set to the most common class label among the samples in the current node (_most_common_label(y)).
    - Return this leaf node.

    b. Find the Best Split (if not stopping) (_best_split function):
    i. Select Features to Consider:
    - If self.n_features is less than the total number of features, randomly select self.n_features to consider for splitting (feat_idxs). Otherwise, consider all features.
    ii. Iterate Through Selected Features and Thresholds:
    - For each selected feature (feat_idx):
    - Get all unique values in that feature column (X_column) to serve as potential thresholds.
    - For each threshold:
    - Temporarily split the current node's data (X_column, y) into a left_child (samples <= threshold) and right_child (samples > threshold).
    - Calculate the Information Gain (or other chosen criterion) for this split (_information_gain(y, X_column, threshold)). This involves:
    - Calculating parent entropy.
    - Calculating weighted average entropy of children.
    - Information Gain = Parent Entropy - Weighted Child Entropy.
    - (Ensure splits don't result in empty children, as your _information_gain returns 0 for such cases).
    iii. Determine Best Split:
    - Keep track of the feature and threshold that result in the best_gain (highest Information Gain).

    c. Create Child Nodes and Recurse:
    - If a best_feature and best_thresh are found (i.e., best_gain is positive, indicating an improvement):
    - Create an Internal Node storing best_feature and best_thresh.
    - Permanently split the current node's data (X, y) into left_dataset and right_dataset using the best_feature and best_thresh (_split function).
    - Recursively call _grow_tree for the left_dataset to create the left child of the current node: left_child = _grow_tree(X_left, y_left, depth + 1).
    - Recursively call _grow_tree for the right_dataset to create the right child of the current node: right_child = _grow_tree(X_right, y_right, depth + 1).
    - Link left_child and right_child to the current internal node.
    - Return the current internal node.
    - Else (if no split improves purity or meets minimum gain criteria, or all samples are identical in features but different in labels and cannot be split further):
    - Create a Leaf Node with the most common class label of the current samples (similar to stopping criteria).
    - Return this leaf node.

    The self.root is set to the node returned by the initial call to _grow_tree(X, y).

II. Testing / Prediction Phase (after training - predict and _traverse_tree methods):

Input New Data:
Take a new, unseen data point x_new (a single sample with all its features).

Traverse the Tree (_traverse_tree(x, node) function):

    Start at the self.root of the trained tree.

    At the current node:
    a. If the node is a is_leaf_node():
    - Return the value (predicted class label) stored in this leaf node.
    b. Else (if it's an internal node):
    - Get the feature index and threshold from the current node.
    - Compare the value of x_new[node.feature] with node.threshold.
    - If x_new[node.feature] <= node.threshold:
    - Recursively call _traverse_tree with x_new and node.left child.
    - Else (x_new[node.feature] > node.threshold):
    - Recursively call _traverse_tree with x_new and node.right child.

Predict Class Label:
The value returned by _traverse_tree for x_new is the predicted class label. For a dataset X_test, this process is repeated for each sample.
'''

import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


