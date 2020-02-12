# %%
'''
# CS178 WINTER 2017 Project
# KODY CHEUNG 85737824
# AARON CHING 28162665
'''

# %%
import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import seaborn as sns
import pandas as pd

import sklearn.model_selection

# import X training points with 15 features and Y training points
X_data = np.genfromtxt("C:\Python35\CS178\Project\X_train.txt",delimiter=None)
Y_data = np.genfromtxt("C:\Python35\CS178\Project\Y_train.txt",delimiter=None)

# import X testing points
X_test = np.genfromtxt("C:\Python35\CS178\Project\X_test.txt",delimiter=None)
Test = X_test.shape[0]

full_ensemble = []

# %%
'''
### 1) K-Nearest Neighbors
'''

# %%
import sklearn.neighbors
import sklearn.decomposition
import sklearn.model_selection
import sklearn.metrics
import sklearn.cross_validation

# create K-nearest neighbor learner

# Different K values for nearest points
nearest = [1, 3, 5, 15, 55, 105]



# Subsampling a smaller part of the data
X_train, X_valid, Y_train, Y_valid = ml.splitData(X_data, Y_data, 0.80)

parameters = {'n_neighbors': nearest}

knearest = sklearn.neighbors.KNeighborsClassifier()

clf = sklearn.model_selection.GridSearchCV(knearest, parameters, cv = 10)

clf.fit(X_train, Y_train)

dimensions = [1,2,3,4,5,6,7,8,9,10]

accuracy = []
params = []



Test = X_test.shape[0]

predict = np.zeros((Test,2))


for d in dimensions:
    svd = sklearn.decomposition.TruncatedSVD(n_components = d)

    X_fit = svd.fit_transform(X_train)
    X_fit_atest = svd.transform(X_valid)

    clf.fit(X_fit, Y_train)

    Kfolds = sklearn.cross_validation.KFold(X_fit_atest.shape[0], n_folds = 10)
    scores = []

    for i,j in Kfolds:
        test_set = X_fit_atest[j]
        test_labels = Y_valid[j]
        scores.append(sklearn.metrics.accuracy_score(test_labels, clf.predict(test_set)))
        
    accuracy.append(scores)
    params.append(clf.best_params_['n_neighbors'])

    test = svd.transform(X_test)
    predict += clf.predict_proba(test)

print(np.mean(accuracy))
predict = predict/10
print(predict)

knn = predict

# np.savetxt('1A-KNN.txt', np.vstack( (np.arange(len(predict)), predict[:,1]) ).T, 
#            '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')


# %%
'''
### 2) SVM Kernel - libSVM but for large amounts of data
'''

# %%
# from sklearn import svm
# from sklearn.decomposition import PCA


X_train, X_valid, Y_train, Y_valid = ml.splitData(X_data, Y_data, 0.80)

#print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn import svm


cc = 100
gamm = 0.00001
svm_learner = svm.SVC(C = cc, gamma = gamm, probability = True)
print("C: ", cc)
print("gamma: ", gamm)
svm_learner.fit(X_train, Y_train)

Yhat = svm_learner.predict(X_valid)
Ythat = svm_learner.predict(X_train)

predictions = svm_learner.predict_proba(X_test)

print('Training Accuracy', np.mean (Ythat == Y_train))
print('Validation Accuracy', np.mean(Yhat == Y_valid))

print(predictions)

np.savetxt('SVM_predict.txt', np.vstack( (np.arange(len(predictions)), predictions[:,1]) ).T, 
            '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')

svmz = predictions

#C value determines how willing you are to misclassify data when
#deciding on the margin (higher the C, the less incorrect points are allowed within the margin)

#its said that a linear function aka no kernel can help prevent overfitting
# where a non linear function normally would, but that never finished running

#small gamma gives low bias and high variance while large gamma will give
#higher bias and low variance



# mu = Y_train.mean()
# dY = Yt - mu
# step = 0.5

# Pt = np.zeros((X_train.shape[0],)) + mu
# Pv = np.zeros((X_valid.shape[0],)) + mu
# Pe = np.zeros((X_test.shape[0],)) + mu

# np.random.seed(0)

# for i in range(0,25):
    
#     svm_learner = svm.SVC(probability = True)
    
#     svm_learner.fit(X_train, Y_train)
    
    
#     Pt += step * svm_learner.predict(X_train)[:,0]
#     Pv += step * svm_learner.predict(X_valid)[:,0]
#     Pe += step * svm_learner.predict(X_test)[:,0]
    
#     dY -= step * svm_learner.predict(X_train)[:,0]
    
#     print(auc(X_valid, Y_valid))
    
    

# Yhat = svm_learner.predict(X_valid)

# predictions = svm_learner.predict_proba(X_valid)

# print('Accuracy', np.mean(Yhat == Y_valid))

# print(predictions)



# # df = pd.DataFrame(X_train)
# # df['rain'] = Y_train
# # sns.pairplot(df, hue='rain', vars = range(14))




# #print("SVM Kernel AUC: {}".format(result))


# %%
'''
### 3) Random forest of treeClassifiers
### MaxDepth = 20, MinLeaf = 4, nFeatures = 10
### 50 bag ensemble
'''

# %%
# dimensions of X_test
m,n = X_test.shape

Test = X_test.shape[0]

# ensemble of classifiers
bags = 50
full_ensemble = [None] * bags
Area_Under_Curve = []

X_train = X_data[0:10000]
X_valid = X_data[10000:20000]

Y_train = Y_data[0:10000]
Y_valid = Y_data[10000:20000]


for i in range(0, bags):
    
    indices = np.floor(m * np.random.rand(m)).astype(int)    # random combination of 100k rows
    
    Xi, Yi = X_data[indices,:], Y_data[indices]              # X and Y indices of those rows
    
    # put the learners in the ensemble
    full_ensemble[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth = 20, minLeaf = 4, nFeatures = 10) 
    

# space for predictions from each model
predict = np.zeros((Test,2))

train = []
valid = []


for i in range(0,bags):
    
    train.append(np.mean(full_ensemble[i].predict(X_train) == Y_train))
    
    valid.append(np.mean(full_ensemble[i].predict(X_valid) == Y_valid))    
    
    # soft predict on each learner in the bag
    predict += full_ensemble[i].predictSoft(X_test)
    
    # get the auc of each learner in the bag
    Area_Under_Curve.append(full_ensemble[i].auc(X_data, Y_data))

# average predictions    
predict = predict/50

print("training error: {}".format(np.mean(train)))
print("validation error: {}".format(np.mean(valid)))

# average auc
result = np.mean(Area_Under_Curve)
print("Random Forest AUC: {}".format(result))


random_forest_bags = predict

print(random_forest_bags)


# np.savetxt('Yhat_ensemble.txt', np.vstack( (np.arange(len(predict)), predict[:,1]) ).T, 
#            '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')

# %%
'''
### 4) Boosted Learner (Adaptive boosting)
'''

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


#GaussianNaiveBayes = GaussianNB()  # low rate of success

decision_tree = DecisionTreeClassifier()

AUC = []

Test = X_test.shape[0]
ultraboost = np.zeros((Test,2))

ensemble = [None] * 25

for i in range(0, 25):
    
    Xtr, Xva, Ytr, Yva = train_test_split(X_data, Y_data, test_size=0.80, random_state = 42)

    
    ensemble[i] = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 15))
    
    ensemble[i].fit(Xtr, Ytr)


    AUC.append(np.mean(ensemble[i].predict(Xva) == Yva))
    
    ultraboost += ensemble[i].predict_proba(X_test)
    
    #print("training error: {}".format(np.mean(ensemble.predict(Xtr) == Ytr)))

ultraboost = ultraboost/25
print(ultraboost)
print(np.mean(AUC))


# np.savetxt('Ultraboost.txt', np.vstack( (np.arange(len(ultraboost)), ultraboost[:,1]) ).T, 
#            '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')
    
#print("AdaBoost AUC: {}".format(result))

# %%
'''
### Gradient Boosting
'''

# %%
from sklearn.ensemble import GradientBoostingClassifier


X_train, X_valid, Y_train, Y_valid = ml.splitData(X_data, Y_data, 0.80)

Test = X_test.shape[0]
predict = np.zeros((Test, 2))


clf = GradientBoostingClassifier(n_estimators=3000, learning_rate=1, max_depth=500
                              ,max_leaf_nodes=50)

clf.fit(X_train, Y_train)

Yhat = (clf.predict(X_valid))
Ythat = (clf.predict(X_train))
gradiboost = clf.predict_proba(X_test)

print(gradiboost)
print('Training Accuracy', np.mean(Ythat == Y_train))
print('Validation Accuracy', np.mean(Yhat == Y_valid))
np.savetxt('GB_predict.txt', np.vstack( (np.arange(len(gradiboost)), gradiboost[:,1]) ).T, 
            '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')

#combines a set of weak learners using a gradient descent-like procedure, outcomes are weighed based on the previous instant
#misclassifed outcomes will have higher weight

#Regression trees are used as weak learners, and their outputs are added together and correct the residuals in the predictions
#(based on a loss function of your choice). A gradient descent procedure is used to minimize the loss when
#adding the trees together.
#We put constraints on the trees to make sure they stay weak. We also weight the predictions of each tree to slow down the
#learning of the algorithm.

#learning rate: impact of each tree on the outcome. Magnitude of changes based
#on the outputs of the trees
#Lower values make the model robust to tree characteristics, and requires more
#trees to model all relations (makes it more expensive)

#n_estimators: number of sequential trees to be modeled
#GBM is pretty robust against overfitting, but still will at some point (should
#tune n_estimators for a particular learning rate)

#max_depth: limits the number of nodes in the tree

# %%
'''
### 6) Neural Network
'''

# %%
from sklearn.neural_network import MLPClassifier

""" solver = 'adam' for large data sets """

Test = X_test.shape[0]
neural_net = np.zeros((Test,2))

train = []
average = []

nFolds = 10;

for iFold in range(nFolds):

    Xtr, Xva, Ytr, Yva = ml.crossValidate(X_data, Y_data, nFolds, iFold)

    neural_network = MLPClassifier(solver = 'adam', random_state = 0)

    neural_network.fit(Xtr, Ytr)

    predict += neural_network.predict_proba(X_test)
    
    train.append(np.mean(neural_network.predict(Xtr) == Ytr))

    average.append(np.mean(neural_network.predict(Xva) == Yva))

    #print(Yhat)

print("training error: {}".format(np.mean(train)))
    
print(np.mean(average))
predict = predict/10
neural_net = predict
print(neural_net)

##np.savetxt('NNET.txt', np.vstack( (np.arange(len(predict)), predict[:,1]) ).T,
##           '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')

#print("Neural Network AUC: {}".format(result))

# %%
'''
### Combination bag of all learners
'''

# %%
# full_ensemble + random_forest_bags + ultraboost + neural_net

# knn
print(knn)
print()

#svm
print(svmz)
print()

# random_forest_bags
print(random_forest_bags)
print()

# ultraboost
print(ultraboost)
print()

# gradiboost
print(gradiboost)
print()


# neural_net
print(neural_net)
print()


full_ensemble = (knn + random_forest_bags + ultraboost + neural_net + svmz + gradiboost)/6

print(full_ensemble)


np.savetxt('6combination.txt', np.vstack( (np.arange(len(full_ensemble)), full_ensemble[:,1]) ).T,
          '%d, %.2f', header = 'ID,Prob1', comments = '', delimiter=',')



# Summation of soft predicts / number of soft predicts

# %%
