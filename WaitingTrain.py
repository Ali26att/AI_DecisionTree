from Decision_Tree import DecisionTree, DTNode
import pandas as pd


data = pd.read_csv('waiting.csv')
x = data.values[:,:-1]
y = data.values[:,-1]

clf = DecisionTree(max_depth=10)
clf.fit(x, y)
clf.tree.show()
clf.tree.save2file('waiting.txt')
