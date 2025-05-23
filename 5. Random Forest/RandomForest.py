'''
Random Forest
To classify (or regress) data points based on their features. It's an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

Estimation: The dataset's pattern is learned by constructing a multitude of decision trees, each trained on a random subset of the data and features. The final prediction is a consensus (voting for classification, averaging for regression) of these individual trees.
Instead of a single separating curve, it creates many decision boundaries (from trees) and combines their outputs.

Key Mechanisms (No single "formula" like logistic regression, but core principles):

    Decision Trees as Base Learners: Each tree partitions the feature space into regions.

    Bagging (Bootstrap Aggregating): Each tree is trained on a random sample of the training data, drawn with replacement. This means some data points might be used multiple times in a single tree's training set, while others might be omitted.

    Feature Randomness (Random Subspace): When splitting a node during the construction of a tree, only a random subset of the total features is considered for finding the best split. This decorrelates the trees.

    Aggregation:

        Classification: The final prediction is the class that receives the most "votes" from all the individual trees.

        Regression: The final prediction is the average of the predictions from all individual trees.

To determine the "goodness" of splits within individual trees, criteria like Gini Impurity or Entropy (for classification) or Mean Squared Error (for regression) are used. The overall model performance is evaluated using metrics like accuracy, precision, recall, F1-score, or Out-of-Bag (OOB) error.

Optimization Technique: Random Forest doesn't use Gradient Descent in the same way as logistic or linear regression to find a global set of parameters. Instead, its "optimization" comes from:

    Greedy Tree Construction: Each decision tree is built greedily, making the best local split at each node based on the chosen criterion (e.g., Gini impurity).

    Ensemble Averaging/Voting: The power comes from combining many (potentially high-variance, low-bias) decorrelated trees, which reduces overall variance and improves generalization.

Key Hyperparameters (analogous to learning rate, but controlling different aspects):

    n_estimators: The number of trees in the forest. More trees generally lead to better performance up to a point, but also increase computation time.

    max_features: The number of features to consider when looking for the best split at each node. A common choice is sqrt(total_features) for classification and total_features / 3 for regression.

    max_depth: The maximum depth of each individual tree. Controls the complexity of the trees.

    min_samples_split: The minimum number of samples required to split an internal node.

    min_samples_leaf: The minimum number of samples required to be at a leaf node.

A good combination of these hyperparameters is crucial for model performance.
Steps:
I. Training Phase:

The goal is to build a "forest" of n_estimators decorrelated decision trees.

    Initialization (Hyperparameters):

        Define n_estimators (e.g., 100).

        Define max_features (e.g., sqrt(number_of_features)).

        Define other tree parameters like max_depth, min_samples_split, min_samples_leaf.

    Iterative Tree Construction (Loop for n_estimators trees):

For each tree t from 1 to n_estimators:


a. Create a Bootstrap Sample:
    - From the original training dataset (N samples, M features), create a new training dataset `D_t` by randomly sampling N instances *with replacement*. This `D_t` will be roughly 2/3 the size of the original dataset in terms of unique instances. The remaining ~1/3 instances are "Out-of-Bag" (OOB) samples for this tree.

b. Grow a Decision Tree (`Tree_t`):
    - Using the bootstrap sample `D_t`.
    - At each node, when deciding on a split:
        i. Randomly select `max_features` from the available M features (without replacement for this specific split).
        ii. Among these `max_features`, find the best feature and split point that maximizes information gain (or minimizes Gini impurity/entropy for classification, or MSE for regression).
    - Grow the tree to its `max_depth` or until other stopping criteria (`min_samples_split`, `min_samples_leaf`) are met. *Individual trees are typically grown deep and are not pruned.*

c. Store the Trained Tree:
    - Add `Tree_t` to the forest.



IGNORE_WHEN_COPYING_START
Use code with caution.
IGNORE_WHEN_COPYING_END

    (Optional but recommended) Out-of-Bag (OOB) Error Estimation:

        For each data point x_i in the original training set:

            Identify all trees in the forest that did not use x_i in their bootstrap sample (these are the OOB trees for x_i).

            Make a prediction for x_i using only these OOB trees (aggregate their predictions).

        The OOB error is the error rate (e.g., misclassification rate) on these OOB predictions. It's a good estimate of the model's generalization error.

II. Testing / Prediction Phase (after training):

Input New Data:
Take a new, unseen data point (or dataset) X_new.

Aggregate Predictions:
For each tree Tree_t in the trained forest:
- Pass X_new through Tree_t to get its prediction ŷ_t_new.

Final Prediction:
- For Classification: The final predicted class for X_new is the class that received the most votes among all ŷ_t_new.
(e.g., if 70 trees predict class 1 and 30 trees predict class 0, the final prediction is 1).
Probabilities can also be estimated by taking the proportion of trees voting for each class.


- **For Regression:** The final predicted value for `X_new` is the average of all `ŷ_t_new`.
  (e.g., `ŷ_new = (1/n_estimators) * Σ ŷ_t_new`).



IGNORE_WHEN_COPYING_START
Use code with caution.
IGNORE_WHEN_COPYING_END

Predict Class Label (if required for classification, based on probabilities):
If using probabilities (e.g., proportion of votes for class 1 is p_1):
If p_1 >= 0.5 → predict 1
If p_1 < 0.5 → predict 0
(This is essentially what majority voting does directly).
'''

# from dataclasses import replace
import numpy as np
from DecisionTree import DecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth,
                         min_sample_split = self.min_samples_split,
                         n_features = self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, X):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions