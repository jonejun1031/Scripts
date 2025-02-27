from hyperopt import tpe, hp
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.random import RandomState
import hyperopt
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV


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


def hyperopt_AdaBoost_objective(params):
    model = ensemble.AdaBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                                        random_state=0)

    clf_ADBT_isotonic = CalibratedClassifierCV(model, cv=5, method='isotonic')

    metric = cross_val_score(clf_ADBT_isotonic, x_train_all, y_train_all,
                             scoring="f1", n_jobs=1)

    return min(-metric)



params_space_AdaBoost = {'n_estimators': hp.choice('n_estimators', range(1,50)),
                        'learning_rate': hp.loguniform('learning_rate', np.log(0.1), np.log(2))}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_AdaBoost_objective,
    space=params_space_AdaBoost,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials,
    rstate=RandomState(0)
)
print('best:')
print(best)


parameters = ['n_estimators', 'learning_rate']
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
    axes[i].set_title("AdaBoost")
    axes[i].set_ylim([0.0, 1.0])
    axes[i].set_ylabel("F1-Score", fontsize=12)
    axes[i].set_xlabel(val, fontsize=12)

plt.show()