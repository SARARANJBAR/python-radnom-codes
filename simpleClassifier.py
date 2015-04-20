




import nltk
from sklearn import cross_validation
import sklearn as sklearn
from sklearn.decomposition import PCA
import numpy as np

# load dataset
dataset = np.loadtxt('/Users/sara/Dropbox/python codes/liverTexture_compact.csv', delimiter=',')
# 4 is the last column, class column , we have to exclude it from the data
features = dataset[:,:4]
features = sklearn.preprocessing.scale(features)

target = dataset[:,4].astype(int)
unique_targets = set(target) 

# instantiate the model
model = PCA(n_components=2)
model.fit(features)
X_PCA = model.transform(features)

#training_set = nltk.classify.apply_features(X_PCA)
cv = cross_validation.KFold(len(X_PCA), n_folds=12, indices=True, shuffle=False)

for traincv, testcv in cv:
    classifier = nltk.NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    print 'accuracy:', nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])