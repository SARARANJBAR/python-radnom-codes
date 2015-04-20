# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:56:40 2015

@author: sara
"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn import svm
import colormap as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import ImgFunctions as Img
import RoiFunctions as ROI
import os



grayScales = 256

def plotData(Datafeatures,Classlabels,featureNames,FeatureClass,axes,title):

    dim = len(featureNames) # number of features
    unique_targets = set(FeatureClass)
    num_classes = len(unique_targets)  # number of classes

  
#    params = {'legend.fontsize': 10, 'legend.linewidth': 2}
#    plt.rcParams.update(params)
#    plt.scatter(Datafeatures[target == 0,0],Datafeatures[target == 0,1],marker='x',c='r')
#    plt.scatter(Datafeatures[target == 1,0],Datafeatures[target == 1,1],marker='>',c='g')
#    plt.scatter(Datafeatures[target == 2,0],Datafeatures[target == 2,1],marker='o',c='b')
#    plt.xlabel(featureNames[0])
#    plt.ylabel(featureNames[1])
#    plt.legend(Classlabels,loc='upper left')
    co = 1
    for r in range(dim-1,0,-1):
        for s in range(r-1,-1,-1): 
             for t,marker,c in zip(xrange(num_classes),">ox","rgb"):
                 # We plot each class on its own to get different colored markers
                 axes.scatter(Datafeatures[target == t,r],Datafeatures[target == t,s],marker=marker,c=c)
#                 axes.xlabel(featureNames[r])
#                 axes.ylabel(featureNames[s])
                 axes.legend(Classlabels,loc='upper left')
                 co+=1
    axes.set_title(title)
#    plt.show()


def calculatePCAofData(data,numComponent):
    
    model = PCA(n_components=2)
    model.fit(data)
    X_PCA = model.transform(data)
#    print "shape of result:", X_PCA.shape
    
    return X_PCA


def CrossValidation(data,target,number_folds,classifier):
    
    loo = cross_validation.LeaveOneOut(n=number_folds)
    scores=[]
    
    for train_index, test_index in loo:
    
         X_train, X_test = data[train_index], data[test_index]
         y_train, y_test = target[train_index], target[test_index]
      
         if (classifier == 'svm'):
             clf = svm.SVC(kernel='linear', C=4).fit(X_train, y_train)
             score = clf.score(X_test, y_test)
             scores.append(score) 
     
    print("classification Results :", scores)


def ConvertDistanceToColor(dist,StartColor,EndColor,StartValue,EndValue):
    
    
    cmap = cm.ColorMap(StartColor, EndColor, StartValue, EndValue)

    if (dist>EndValue):
        
        return cmap.__getitem__(EndValue)
        
    elif (dist<StartValue):
        
        return cmap.__getitem__(StartValue)
        
    else:
        
       return cmap.__getitem__(dist)


def BuildSVMmodel(X,Y,C,kernel,gamma):

    clf = svm.SVC(kernel = kernel,  gamma=gamma, C=C )
    clf.fit(X, Y)
    return clf
   
def loadImage(root, fnImg):
    
    inputImage = Img.readImage(root, fnImg)
    return inputImage
    

def loadROI(rootFnROI):
    
     rect = ROI.parseRectROIFile( rootFnROI )
     power2rect = convertROIToPowerOf2( rect )
     return power2rect
     
def extractSubImg(img,rectCoords):
    
    subImage = Img.extractRect(img,rectCoords)
    subImage = Img.scaleIntensity( subImage, 0, grayScales )
    subImage = np.rint( subImage ).astype( np.uint8 )
    
    plt.imshow(inputImage,cmap=plt.cm.gray)
    plt.title("subImage scaled")
    plt.show()    
    
    
    
def calculatePixelWiseTextureFeatures(path, imgName, roiName):
    
    rootFnROI = os.path.join(path, roiName)  
    rectRoi   = loadROI( rootFnROI )
    image = loadImage(path,imgName)
    
    subImage =  extractSubImg( inputImage, rect )
    
    
    
    
     
def plot_Boundary(axis,svm_model,X,Y):
    
#    fig = plt.figure()
   
    X = StandardScaler().fit_transform(X)
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print Z.shape
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    
    axis.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.bone)

    unique_targets = set(Y)
    num_classes = len(unique_targets)  # number of classes
    
    for t,c in zip(xrange(num_classes),"rgb"):
        axis.scatter(X[target == t,1],X[target == t,0],marker='o',c=c)
                

#    axis.axis('off')
    plt.xlabel('PCA_feature1')
    plt.ylabel('PCA_feature2')
    plt.title('svm decision boundary' )
    plt.show()
    
def CalculateDistanceToBoundary(svmModel,data_samples):
   
    distances = []
    distances = np.append(distances, svm.SVC.decision_function(svmModel,data_samples))
    return distances
#==============================================================================
# main
#==============================================================================

dataset = np.loadtxt('/Users/sara/Dropbox/python codes/liverTexture_compact.csv', delimiter=',')
features = dataset[:,:4]
features = scale(features)
target = dataset[:,4].astype(int)
feature_names = np.array(['Normalized_Dost1','Normalized_Dost3','Normalized_Dost4','Normalized_LogHist skewness Sigma=4.0'])
if(len(feature_names)!=np.shape(features)[1]):
    print("EROR! feature names array length is different from number of features given")
labels = np.array(['cirrhotic','fatty','normal'])
#plotData(features,labels,feature_names,target,(10,10))



# PCA 
pca_data = calculatePCAofData(features,2)
pca_feature_names = np.array(['PCA_feature1','PCA_feature2'])
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plotData(pca_data,labels,pca_feature_names,target,axes[0],'data')
#CrossValidation(pca_data,target,12,'svm')


x_samples1 = pca_data[target == 0,:] 
x_samples2 = pca_data[target == 1,:] 
x_samples3 = pca_data[target == 2,:] 

#kernel can be 'poly','rbf','linear'
svm_model = BuildSVMmodel(pca_data,target,1.0,'rbf',0.7)

plot_Boundary(axes[1],svm_model,pca_data,target)


#==============================================================================
# color the whole plot
#x = np.linspace(-5,5,20)
#y = np.linspace(-5,5,20)
#
#
#start_value,end_value = -3, 3
#
#fig = plt.figure()
#ax = fig.add_subplot(111 , projection='3d')
#for i in x: 
#    for j in y:
#        val = CalculateDistanceToBoundary(svm_model,np.array([i,j]))
#        col = ConvertDistanceToColor(val, cm.Color.ROYALBLUE, cm.Color.RED, start_value, end_value)
#        p = ax.scatter(i,j ,val, c=col, marker='o')
#
#plt.show()

#=============================================================================

