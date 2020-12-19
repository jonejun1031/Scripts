from hyperopt import tpe, hp, rand, anneal
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.random import RandomState
import hyperopt
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def GetDataByPandas():
    data = pd.read_csv("E:\\My Coding\\Scripts\\Data for hyperopt\\lipinski_ecfp6_1024.csv")
    y = np.array(data['class'])
    X = np.array(data.drop("class", axis=1))
    columns = np.array(data.columns)

    return X, y, columns


X, y, Names = GetDataByPandas()

# split data to [[0.75,0.25],0.2]
x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.20, random_state=0)
#x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.20, random_state=0)


def hyperopt_SVM_objective(params):
    model = SVC(C=params['C'], kernel='rbf', degree=3, gamma=params['gamma'], coef0=0.0, shrinking=True,
                probability=True, tol=0.001, cache_size=200, class_weight='balanced', verbose=False,
                max_iter=-1, random_state=0)

    metric = cross_val_score(model, x_train_all, y_train_all, cv=5,
                             scoring="f1", n_jobs=1)

    return min(-metric)



params_space_SVM = {'C': hp.loguniform('C', np.log(1), np.log(1000)),
                    'gamma': hp.loguniform('gamma', np.log(0.0001), np.log(1))}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_SVM_objective,
    space=params_space_SVM,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials,
    rstate=RandomState(0)
)
print('best:')
print(best)


parameters = ['C', 'gamma']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(25, 10))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    axes[i].scatter(
        xs,
        ys,
        s=30,
        linewidth=1,
        alpha=0.8,
        c=cmap(float(i) / len(parameters)))
    axes[i].set_title("SVM_rbf")
    axes[i].set_ylim([0.0, 1.0])
    axes[i].set_ylabel("F1-Score", fontsize=12)
    axes[i].set_xlabel(val, fontsize=12)

plt.show()