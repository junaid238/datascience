import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import metrics
# for spliting data
from sklearn.cross_validation import train_test_split
# importing neighbors from sklearn 
from sklearn import neighbors
from mlxtend.plotting import plot_decision_regions


# load the iris data set into iris object 
iris=load_iris()

# make X as data part with only first two features and Y as target part 
X=iris.data[:, [0, 2]]
y=iris.target

# split data into train and test for x and y features in 70 % 30 % format
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# getting plots for entire data points of iris
# plt.plot(X,y)
# plt.show()

# checking the split for train and test data ( 0.3 )
print(" Train data length of X feature " + str(len(X_train)))
print(" Train data length of y feature " + str(len(y_train)))
print(" Test data length of X feature " + str(len(X_test)))
print(" Test data length of y feature " + str(len(y_test))) 

# making a KNN classifier using neighbors
# single K value for classification 
# k= 10 
# clf = neighbors.KNeighborsClassifier(k, weights='uniform')
# fit the train data into classifier
# clf.fit(X_train, y_train)
# print("plotting regions")
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.title('K = %i Graph' %(k))
# plot_decision_regions(X_train,y_train, clf=clf)
# plt.show()

# # multiple K values for classification using uniform weights
# k_range= [1,15,30,45,70]
# # to store the accuracy score 
# scores=[]
# for k in k_range :
# 	print("making clf")
# 	# making a classifier using k list and uniform weights
# 	clf = neighbors.KNeighborsClassifier(k, weights='uniform')
# 	print("fitting clf")
# 	# fitting training data 
# 	clf.fit(X_train,y_train)
# 	# prediction of testing data of data features
# 	y_pred=clf.predict(X_test)
# 	# calculating the accuracy using metrics
# 	scores.append(metrics.accuracy_score(y_test,y_pred))
# 	# plotting regions of predictions 
# 	print("plotting regions")
# 	plt.xlabel('sepal length [cm]')
# 	plt.ylabel('petal length [cm]')
# 	plt.title('K = %i Graph uniform weights' %(k))
# 	# plotting entire training data using X_train and y_train
# 	# plot_decision_regions(X_train,y_train, clf=clf)
# 	# plotting tetsing data and predicted values of y using X_test and y_pred
# 	plot_decision_regions(X_test,y_pred, clf=clf)
# 	plt.show()

# print(scores)

# multiple K values for classification using distance weights
k_range= [1,15,30,45,70]
# to store the accuracy score 
scores=[]
for k in k_range :
	print("making clf")
	# making a classifier using k list and distance weights
	clf = neighbors.KNeighborsClassifier(k, weights='distance')
	print("fitting clf")
	# fitting training data 
	clf.fit(X_train,y_train)
	# prediction of testing data of data features
	y_pred=clf.predict(X_test)
	# calculating the accuracy using metrics
	scores.append(metrics.accuracy_score(y_test,y_pred))
	# plotting regions of predictions 
	print("plotting regions")
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.title('K = %i Graph distance weights' %(k))
	# plotting entire training data using X_train and y_train
	# plot_decision_regions(X_train,y_train, clf=clf)
	# plotting tetsing data and predicted values of y using X_test and y_pred
	plot_decision_regions(X_test,y_pred, clf=clf)
	plt.show()

print(scores)

error = []
for score in scores:
	error.append(1-score)
# plotting error curve for various values of k 
plt.plot(k_range,error)
# plt.plot(k_range,error)
plt.savefig("dataError.png")
plt.show()