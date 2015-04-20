# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:57:11 2015

@author: sara
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

def decision_boundary(x_vec, mu_vec1, mu_vec2):
    g1 = (x_vec-mu_vec1).T.dot((x_vec-mu_vec1))
    g2 = 2*( (x_vec-mu_vec2).T.dot((x_vec-mu_vec2)) )
    boundary =  g1 - g2
    f, ax = plt.subplots(figsize=(7, 7))
    c1, c2 = "#3366AA", "#AA3333"
    ax.scatter(*x1_samples.T, c=c1, s=40)
    ax.scatter(*x2_samples.T, c=c2, marker="D", s=40)
    x_vec = np.linspace(*ax.get_xlim())
    ax.contour(x_vec, x_vec,boundary,levels=[0], cmap="Greys_r")

def plot_svm_boundary(x1_samples,x2_samples,C,kernel,gamma):
    
    fig = plt.figure()
    plt.scatter(x1_samples[:,0],x1_samples[:,1], marker='+')
    plt.scatter(x2_samples[:,0],x2_samples[:,1], c= 'green', marker='o')

    X = np.concatenate((x1_samples,x2_samples), axis = 0)
    Y = np.array([0]*100 + [1]*100)
    
     
    clf = svm.SVC(kernel = kernel,  gamma=gamma, C=C )
    clf.fit(X, Y)
    
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)




#==============================================================================
# main    
#==============================================================================
mu_vec1 = np.array([0,0])
cov_mat1 = np.array([[2,0],[0,2]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
print x1_samples.shape
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([1,2])
cov_mat2 = np.array([[1,0],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1,2).T





#decision_boundary(x_vec,mu_vec1,mu_vec2)

gamma= 0.7
C = 1.0     # SVM regularization parameter
kernel = 'rbf' # Svm kernel , can also be 'linear'


plot_svm_boundary(x1_samples,x2_samples,4,'rbf',.7)