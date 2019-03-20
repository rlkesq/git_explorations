#%%
import pandas as pd 
import numpy as np 
import sklearn
from sklearn import datasets, model_selection
from sklearn.model_selection import  train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
iris = datasets.load_iris()
print('iris is of shape: ', iris.data.shape)
#%%
''' Shifting Gears: Let's explore support vector machines
We start with a linear model (where the data can be separated in two
dimensions) using the 'linear' kernel.
Then we will move to higher dimensional analysis using the  kernel SVM'''
clf = svm.LinearSVC(max_iter=5000)
x = iris.data
# print(x[1:5])
y = iris.target
print(x.shape, y.shape)
clf.fit(x,y)
print(clf.predict([[ 6.0,  4.0,  3.6,  0.35]]))
print(clf.coef_)
#
note_df = pd.read_csv('/users/richardkornblith/data_science/git_explore/bill_authentication.csv')
print('\n Bank Note Information: \n\n',note_df.info())
#%%
xx = note_df.drop('Class', axis=1)
xx.info()
yy = note_df['Class']
X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=.20)
print(X_train.shape, y_train.shape)
from sklearn.svm import SVC
svlclassifier = SVC(kernel = 'linear')
svlclassifier.fit(X_train, y_train)
y_pred = svlclassifier.predict(X_test)
print(type(y_pred))
print (confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
#%%
'''Now on to the higher dimensions!'''
url = url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names = colnames)
X =irisdata.drop(['Class'], axis=1) 
y = irisdata['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)
#%%
'''There are three types of kernels available using the SVM higher dimensional machines.
These are the polynomial, the Gaussian and the Sigmoid kernels.  With the first
two, there are various options: for the polynomial, one must specify the order;
for the Gaussian, one can specify 'rbf'.  Note that the Sigmoid kernel is relevant
only where the classification is to be binary.'''


#%%
svpoly = SVC(kernel='poly', degree = 7, gamma = 'auto')
svpoly.fit(X_train, y_train)
y_predpoly = svpoly.predict(X_test)
print (confusion_matrix(y_test, y_predpoly))
print (classification_report(y_test,y_predpoly))

#%%
svgauss = SVC(kernel = 'rbf', gamma = 'scale')
svgauss.fit (X_train, y_train)
y_predgauss = svgauss.predict(X_test)
print (confusion_matrix(y_test, y_predgauss))
print (classification_report(y_test, y_predgauss))

#%%
# For completeness only; see note, above re binary classification
svsigmoid = SVC(kernel = 'sigmoid', gamma = 'scale')
svsigmoid.fit(X_train, y_train)
y_predsigmoid = svsigmoid.predict(X_test)
print(confusion_matrix(y_test, y_predsigmoid))
print (classification_report(y_test, y_predsigmoid))

#%%
