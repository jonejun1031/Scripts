from hyperopt import fmin, tpe, hp, rand, anneal
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.random import RandomState
import hyperopt
from sklearn.ensemble import RandomForestClassifier
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


def hyperopt_RF_objective(params):
    model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], min_samples_split=2,
                                   max_features=params['max_features'], max_leaf_nodes=None, bootstrap=False, oob_score=False,
                                   n_jobs=-1, random_state=0)

    metric = cross_val_score(model, x_train_all, y_train_all, cv=5,
                             scoring="f1", n_jobs=-1)

    return min(-metric)



params_space_RF = {'n_estimators': hp.choice('n_estimators', range(1,200)),
                   'criterion': hp.choice('criterion', ['gini', 'entropy']),
                   'max_features': hp.choice('max_features', ['log2', 'sqrt'])}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_RF_objective,
    space=params_space_RF,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials,
    rstate=RandomState(0)
)
print('best:')
print(best)


parameters = ['n_estimators', 'criterion', 'max_features']
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
    axes[i].set_title("RF")
    axes[i].set_ylim([0.0, 1.0])
    axes[i].set_ylabel("F1-Score", fontsize=12)
    axes[i].set_xlabel(val, fontsize=12)

plt.show()