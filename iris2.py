from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


iris = load_iris()
print "FEATURES OF IRIS"
print (iris.feature_names)
print "IRIS DATASET"
print(iris.data)
print "TARGET LOCATION"
print(iris.target)
print "TARGET NAMES"
print(iris.target_names)
print(iris.target_names[2])



knn = KNeighborsClassifier(n_neighbors=1)
X = iris.data
Y = iris.target
knn.fit(X,Y)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_neighbors=1, p=2, weights='uniform')
print("testing:")
test_input=[3,5,4,2]
test_input = np.array(test_input).reshape(1, -1)
species_id = knn.predict(test_input)
print iris.target_names[species_id]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
x = Y_test
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, Y_train)
y = knn.predict(X_test)
print("training the data")
print(y)
print iris.target_names[y]

print('accuracy is',accuracy_score(x,y))

print("predicting:")
print("enter the no. of iris flowers to be predicted")
n=input()
for i in range(1,n+1):
   num=raw_input('enter input seperated by space')
   num_arr=num.split(" ")
   num_arr = np.array(num_arr).reshape(1, -1)
   species_id = knn.predict(num_arr)
   print iris.target_names[species_id]

