# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:12:12 2015

@author: sara
"""

from matplotlib import pyplot as plt
#from sklearn.datasets import load_iris
import numpy as np
from xlrd import *
import sklearn as sklearn
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from numpy import *
#data = load_iris()
#features = data['data']
#feature_names = data['feature_names']
#target = data['target']


dataset = np.loadtxt('/Users/sara/Dropbox/python codes/liverTexture_compact.csv', delimiter=',')
features = dataset[:,:4]
target = dataset[:,4].astype(int)
feature_names = np.array(['Normalized_Dost1','Normalized_Dost3','Normalized_Dost4','Normalized_LogHist skewness Sigma=4.0'])


features = sklearn.preprocessing.scale(features)
labels = np.array(['cirrhotic','fatty','normal'])
n = len(feature_names)
unique_targets = set(target) 
number_types = len(unique_targets)

plt.figure(figsize = (10,10))
co = 1
params = {'legend.fontsize': 10,
          'legend.linewidth': 2}
plt.rcParams.update(params)

for r in range(n-1,0,-1):
     for s in range(r-1,-1,-1): 

         plt.subplot(2,n-1,co)
         
         for t,marker,c in zip(xrange(number_types),">ox","rgb"):
             # We plot each class on its own to get different colored markers
             h = plt.scatter(features[target == t,r],features[target == t,s],marker=marker,c=c)
             plt.xlabel(feature_names[r])
             plt.ylabel(feature_names[s])
             plt.legend(labels,loc='upper left')
         co+=1


plt.show()


# instantiate the model
model = PCA(n_components=2)
model.fit(features)
# transform the data to two dimensions
X_PCA = model.transform(features)
print "shape of result:", X_PCA.shape

# plot the results along with the labels
plt.figure(2)
params = {'legend.fontsize': 12,
          'legend.linewidth': 2}
plt.scatter(X_PCA[:, 0], X_PCA[:, 1],c=target)
for t,c in zip(xrange(number_types),"rgb"):
             # We plot each class on its own to get different colored markers
             h = plt.scatter(X_PCA[target==t, 0], X_PCA[target==t, 1],marker="o",c=c)
plt.xlabel('PCA_feature1')
plt.ylabel('PCA_feature2')
plt.legend(labels,loc='upper right')
plt.show()

#print(X_PCA.shape,target.shape)

##==============================================================================
#
#scores = list()
##==============================================================================
# 
loo = cross_validation.LeaveOneOut(n=12)
n=len(target)
scores=[]
mean_scores = [] 

for train_index, test_index in loo:

     X_train, X_test = X_PCA[train_index], X_PCA[test_index]
     y_train, y_test = target[train_index], target[test_index]
  
     clf = SVC(kernel='linear', C=4).fit(X_train, y_train)
     score = clf.score(X_test, y_test)
     scores.append(score) 
 
print("Accuracy: %0.2f" % mean(scores))
mean_scores.append(mean(scores))
##==============================================================================
#
