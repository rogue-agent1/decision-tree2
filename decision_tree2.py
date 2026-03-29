#!/usr/bin/env python3
"""Decision tree classifier with entropy-based splitting (ID3)."""
import sys, math
from collections import Counter

def entropy(labels):
    n = len(labels); counts = Counter(labels)
    return -sum((c/n) * math.log2(c/n) for c in counts.values() if c > 0)

def info_gain(data, labels, feature_idx):
    total_ent = entropy(labels); n = len(data)
    values = set(row[feature_idx] for row in data)
    weighted = 0
    for v in values:
        subset_labels = [labels[i] for i, row in enumerate(data) if row[feature_idx] == v]
        weighted += (len(subset_labels) / n) * entropy(subset_labels)
    return total_ent - weighted

class TreeNode:
    def __init__(self, feature=None, value=None, label=None, children=None):
        self.feature, self.value, self.label = feature, value, label
        self.children = children or {}

def build_tree(data, labels, features, depth=0, max_depth=10):
    if len(set(labels)) == 1: return TreeNode(label=labels[0])
    if not features or depth >= max_depth: return TreeNode(label=Counter(labels).most_common(1)[0][0])
    gains = [(info_gain(data, labels, f), f) for f in features]
    _, best = max(gains, key=lambda x: x[0])
    node = TreeNode(feature=best)
    values = set(row[best] for row in data)
    remaining = [f for f in features if f != best]
    for v in values:
        subset_data = [row for i, row in enumerate(data) if row[best] == v]
        subset_labels = [labels[i] for i, row in enumerate(data) if row[best] == v]
        if subset_data:
            node.children[v] = build_tree(subset_data, subset_labels, remaining, depth+1, max_depth)
        else:
            node.children[v] = TreeNode(label=Counter(labels).most_common(1)[0][0])
    node._default = Counter(labels).most_common(1)[0][0]
    return node

def predict(tree, sample):
    if tree.label is not None: return tree.label
    val = sample[tree.feature] if isinstance(sample, list) else sample.get(tree.feature)
    child = tree.children.get(val)
    if child: return predict(child, sample)
    return getattr(tree, '_default', None)

def main():
    if len(sys.argv) < 2: print("Usage: decision_tree2.py <demo|test>"); return
    if sys.argv[1] == "test":
        assert abs(entropy(["a","a","b","b"]) - 1.0) < 0.01
        assert entropy(["a","a","a"]) == 0
        # Play tennis dataset (simplified)
        data = [["sunny","hot"],["sunny","cool"],["rain","hot"],["rain","cool"],["overcast","hot"]]
        labels = ["no","yes","no","yes","yes"]
        tree = build_tree(data, labels, [0, 1])
        assert tree.feature is not None
        p = predict(tree, ["sunny","cool"]); assert p == "yes"
        p2 = predict(tree, ["overcast","hot"]); assert p2 == "yes"
        # Single class
        tree2 = build_tree([[1],[2]], ["a","a"], [0])
        assert tree2.label == "a"
        # Info gain
        g = info_gain(data, labels, 0); assert g > 0
        print("All tests passed!")
    else:
        data = [["yes","no"],["yes","yes"],["no","yes"],["no","no"]]
        labels = ["A","A","B","B"]
        tree = build_tree(data, labels, [0, 1])
        print(f"Root splits on feature {tree.feature}")

if __name__ == "__main__": main()
