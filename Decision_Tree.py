import numpy as np
from collections import Counter
from treelib import Node, Tree

class DTNode:
    def __init__(self, feature=None, min=None, max=None, gain=None, entrp=None, children=None,*,value=None):
        self.feature = feature
        self.min = min
        self.max = max
        self.gain = gain
        self.entrp = entrp
        self.children = children
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, splitter=3, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None
        self.splitter = splitter
        self.tree = Tree()
        self.n_nodes = 0

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0, parent=None):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            self._create_node(None,None,None,leaf_value,parent)
            return DTNode(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, gain, children_idx, max, min = self._best_split(X, y, feat_idxs)
        entrp = self._entropy(y)
        # create child nodes
        parent = self._create_node(best_feature,entrp,gain,None,parent)
        children = []
        for child in children_idx:
            children.append(self._grow_tree(X[child, :], y[child], depth+1,parent))
        return DTNode(best_feature,min, max, gain, entrp, children)

    def _create_node(self,feature=None,entrp=None,gain=None,value=None,parent=None):
        if parent is None:
            if value is None:
                _printer = "Feature: " + str(feature) + " Entropy: " + str(entrp) + " Gain: " + str(gain)
                parent = 0
                self.tree.create_node(_printer,self.n_nodes,parent=None)
                self.n_nodes = self.n_nodes +1
                return self.n_nodes-1
            else:
                _printer1 = "It is a leaf with the value: " + str(value)
                parent = 0
                self.tree.create_node(_printer1,self.n_nodes,parent=None)
                self.n_nodes = self.n_nodes +1
                return self.n_nodes-1

        else:
            if value is None:
                _printer = "Feature: " + str(feature) + " Entropy: " + str(entrp) + " Gain: " + str(gain)
                self.tree.create_node(_printer,self.n_nodes,parent=parent)
                self.n_nodes = self.n_nodes +1
                return self.n_nodes-1
            else:
                _printer1 = "It is a leaf with the value: " + str(value)
                self.tree.create_node(_printer1,self.n_nodes,parent=parent)
                self.n_nodes = self.n_nodes +1
                return self.n_nodes-1

                
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx = None
        best_children = None
        best_max= None
        best_min = None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            max = np.max(X_column)
            min = np.min(X_column)
            gain , children = self._information_gain(y, X_column, min,max)
            
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                best_children = children
                best_max = max
                best_min = min


        return split_idx, best_gain, best_children, best_max, best_min


    def _information_gain(self, y, X_column, min, max):
        if max == min:
            return 0,[]
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        children1 = self._split(X_column, min, max)

        if len(children1) == 0:
            return 0

        children = []
        for elem in children1:
            if elem.any():
                children.append(elem)
        # calculate the weighted avg. entropy of children
        n = len(y)
        children_entropy = 0
        for child in children:
            child_length = len(child)
            child_entropy = self._entropy(y[child])
            children_entropy = children_entropy + ((child_length/n) * child_entropy)

        # calculate the IG
        information_gain = parent_entropy - children_entropy
        return information_gain , children

    def _split(self, X_column, min, max):
        rng = (max-min)/self.splitter
        children =[]
        i = min
        while(i< max and i >= min):
            children.append(np.argwhere(np.logical_and(i <= X_column,X_column < i+rng)).flatten()) 
            i=i+rng
        children.append(np.argwhere(np.logical_and(i <= X_column,X_column < i+rng)).flatten()) 
        return children

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        rng = (node.max-node.min)/self.splitter
        n=0
        i = node.min
        while(i< node.max and i >= node.min):
            if x[node.feature] >= i or x[node.feature] < i+rng:
                return self._traverse_tree(x, node.children[n])
            i=i+rng
            n=n+1
