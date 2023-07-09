from sklearn.model_selection import train_test_split
import numpy as np
from Decision_Tree import DecisionTree, DTNode
import pandas as pd

data = pd.read_csv('diabetes.csv')
x = data.values[:,:-1]
y = data.values[:,-1]
y = np.array(y, dtype="int64")
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=1234
)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
clf.tree.show()
clf.tree.save2file('diabetes.txt')

predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)
