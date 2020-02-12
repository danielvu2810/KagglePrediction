# %%
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# %%
X = np.genfromtxt("data/X_train.txt", delimiter=None)
Y = np.genfromtxt("data/Y_train.txt", delimiter=None)
xte = np.genfromtxt("data/X_test.txt", delimiter=None)

# %%
Xtr = X[:190000]
Ytr = Y[:190000]
Xva = X[190000:]
Yva = Y[190000:]

# %%
#Gradient boost
gb=GradientBoostingClassifier(n_estimators=500,max_depth=12,min_samples_split=12, min_samples_leaf=5)
gb.fit(Xtr,Ytr)



# %%
Ytr_score = gb.predict_proba(Xtr).T[1]
Y_score = gb.predict_proba(Xva).T[1]


print("Trainning auc",roc_auc_score(Ytr,Ytr_score))
print("Validation auc",roc_auc_score(Yva,Y_score))


# %%
from sklearn import *
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(
    X, Y, test_size=0.2, random_state=42)

# %%
#Random forest
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(
n_estimators=1000, min_samples_leaf=4,max_depth= 50,oob_score=True)
classf = random_forest_classifier.fit(x_train, y_train)
Y_score = classf.predict_proba(x_validation).T[1]
Ytr_score = classf.predict_proba(x_train).T[1]


# %%
print("Training AUC: {}".format(roc_auc_score(y_train,Ytr_score)))
print("Validation AUC: {}".format(roc_auc_score(y_validation,Y_score)))

# %%
#Ada boost
ab = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=23,max_features=4,min_samples_split=32), n_estimators=800,learning_rate=0.001)
ab.fit(Xtr,Ytr)


#这里是 estimator 关系不大，100， 80 都差不多
# 10是 0.95 0.768
# 50 是 0.98 0.786
# Trainning auc 0.9901118657901069  120
# Validation auc 0.7886719734892589
# 150
# 0.001
#
Y_ptr = ab.predict_proba(Xtr).T[1]
Y_pr = ab.predict_proba(Xva).T[1]


print("Trainning auc",roc_auc_score(Ytr,Y_ptr))
print("Validation auc",roc_auc_score(Yva,Y_pr))

# %%
from sklearn.ensemble import VotingClassifier

# %%
blend = VotingClassifier(estimators = [('rf',random_forest_classifier),
                                      ("ab",ab),
                                      ("gb",gb)],voting = "soft",
                        weights = [1,1,2])

# %%
blend.fit(Xtr,Ytr)


# %%
Y_ptr = blend.predict_proba(Xtr).T[1]
Y_pr = blend.predict_proba(Xva).T[1]

print("Trainning auc",roc_auc_score(Ytr,Y_ptr))
print("Validation auc",roc_auc_score(Yva,Y_pr))

# %%
Y_ptr = blend.predict(Xtr)
Y_pr = blend.predict(Xva)

print("Trainning auc",roc_auc_score(Ytr,Y_ptr))
print("Validation auc",roc_auc_score(Yva,Y_pr))

# %%
Xte = np.genfromtxt('data/X_test.txt', delimiter=None)
Yte = np.vstack((np.arange(Xte.shape[0]), blend.predict_proba(Xte).T[1])).T

np.savetxt('Y_submit_blend.txt',Yte,'%d, %.2f',header='ID,Prob1',comments='',delimiter=',')

# %%
'''
kaggle result AUC:0.799
'''

# %%
