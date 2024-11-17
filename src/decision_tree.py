"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        X = X.to_numpy()
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # raise NotImplementedError
        # Check stopping conditions
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        # Stopping condition
        if depth >= self.max_depth or num_classes == 1 or num_samples == 0:
            leaf_value = self._majority_class(y)
            return {"leaf": leaf_value}
        
        # Find best split
        feature_index, threshold = find_best_split(X, y)
        if feature_index is None:
            leaf_value = self._majority_class(y)
            return {"leaf": leaf_value}
        
        # Split the data
        left_idx = X[:, feature_index] < threshold
        right_idx = X[:, feature_index] >= threshold
        left_child = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {"feature_index": feature_index, "threshold": threshold, "left": left_child, "right": right_child}

    def _majority_class(self, y):
        # Returns the majority class in y
        values, counts = np.unique(y, return_counts=True)
        majority_class = values[np.argmax(counts)]
        return majority_class

    def predict(self, X):
        # raise NotImplementedError
        X = X.to_numpy()
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        # raise NotImplementedError
        # Traverse the tree recursively for prediction
        if "leaf" in tree_node:
            return tree_node["leaf"]
        
        feature_index = tree_node["feature_index"]
        threshold = tree_node["threshold"]
        
        if x[feature_index] < threshold:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    # raise NotImplementedError
    left_idx = X[:, feature_index] < threshold
    right_idx = ~left_idx
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]


# Find the best split for the dataset
def find_best_split(X, y):
    # raise NotImplementedError
    best_feature, best_threshold, best_info_gain = None, None, -np.inf
    n_samples, n_features = X.shape
    current_entropy = entropy(y)

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, y_left, _, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # Compute information gain
            left_entropy = entropy(y_left)
            right_entropy = entropy(y_right)
            info_gain = current_entropy - (
                len(y_left) / n_samples * left_entropy + len(y_right) / n_samples * right_entropy
            )

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold


def entropy(y):
    # raise NotImplementedError
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))
