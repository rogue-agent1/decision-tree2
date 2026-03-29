#!/usr/bin/env python3
"""decision_tree2 - Decision tree classifier with entropy and Gini impurity."""
import sys, math
from collections import Counter

def entropy(labels):
    n = len(labels)
    if n == 0: return 0
    counts = Counter(labels)
    return -sum((c/n) * math.log2(c/n) for c in counts.values() if c > 0)

def gini(labels):
    n = len(labels)
    if n == 0: return 0
    counts = Counter(labels)
    return 1 - sum((c/n)**2 for c in counts.values())

class DTNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class DecisionTree:
    def __init__(self, max_depth=10, criterion="entropy"):
        self.max_depth = max_depth
        self.criterion = entropy if criterion == "entropy" else gini
        self.root = None

    def _best_split(self, X, y):
        best_gain = -1
        best_feat = None
        best_thresh = None
        parent_imp = self.criterion(y)
        n = len(y)
        ndim = len(X[0])
        for f in range(ndim):
            values = sorted(set(X[i][f] for i in range(n)))
            for i in range(len(values) - 1):
                thresh = (values[i] + values[i+1]) / 2
                left_y = [y[j] for j in range(n) if X[j][f] <= thresh]
                right_y = [y[j] for j in range(n) if X[j][f] > thresh]
                if not left_y or not right_y:
                    continue
                gain = parent_imp - (len(left_y)/n * self.criterion(left_y) + len(right_y)/n * self.criterion(right_y))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh
        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < 2:
            return DTNode(label=Counter(y).most_common(1)[0][0])
        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            return DTNode(label=Counter(y).most_common(1)[0][0])
        left_idx = [i for i in range(len(y)) if X[i][feat] <= thresh]
        right_idx = [i for i in range(len(y)) if X[i][feat] > thresh]
        left = self._build([X[i] for i in left_idx], [y[i] for i in left_idx], depth+1)
        right = self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth+1)
        return DTNode(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(X, y, 0)

    def _predict_one(self, node, x):
        if node.label is not None:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, x):
        return self._predict_one(self.root, x)

def test():
    assert abs(entropy(["A","A","B","B"]) - 1.0) < 0.01
    assert abs(gini(["A","A","B","B"]) - 0.5) < 0.01
    assert entropy(["A","A","A"]) == 0
    X = [[0,0],[1,0],[0,1],[1,1],[5,5],[6,5],[5,6],[6,6]]
    y = [0,0,0,0,1,1,1,1]
    dt = DecisionTree(max_depth=5)
    dt.fit(X, y)
    assert dt.predict([0.5, 0.5]) == 0
    assert dt.predict([5.5, 5.5]) == 1
    correct = sum(1 for i in range(len(X)) if dt.predict(X[i]) == y[i])
    assert correct == len(X)
    dt2 = DecisionTree(max_depth=5, criterion="gini")
    dt2.fit(X, y)
    assert dt2.predict([0, 0]) == 0
    print("All tests passed!")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("decision_tree2: Decision tree classifier. Use --test")
