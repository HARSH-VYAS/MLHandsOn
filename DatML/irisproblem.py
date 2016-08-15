#iris problem
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image 
from sklearn.externals.six import StringIO 
import pydot

iris = load_iris()

# print the first row.
print iris.feature_names
print iris.target_names

test_idx = [0,50,100]
# Training data
train_target = np.delete(iris.target,test_idx)
train_data =  np.delete(iris.data,test_idx, axis=0)

#Testing data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print clf.predict(test_data)

 
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
