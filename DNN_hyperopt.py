import hyperopt
from hyperopt import tpe, hp
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import time


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def GetDataByPandas():
    data = pd.read_csv("E:\\My Coding\\Scripts\\Data for hyperopt\\lipinski_ecfp6_1024.csv")
    y = np.array(data['class'])
    X = np.array(data.drop("class", axis=1))
    columns = np.array(data.columns)

    return X, y, columns


X, y, Names = GetDataByPandas()

# split data to [[0.75,0.25],0.2]
x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.20, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=0)


def hyperopt_DNN_objective(params):
    model = tf.keras.models.Sequential()
    model.add(Dense(1024, input_dim=1024, kernel_initializer='lecun_uniform',
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.005), name='Dense_1'))
    model.add(Activation(params['activation_type']))
    model.add(Dropout(params['dropout']))

    for idx in range(params['n_layers'] - 1):
        model.add(Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=0.005),
                        kernel_initializer='lecun_uniform'))
        model.add(Activation(params['activation_type']))
        model.add(Dropout(params['dropout']))

    model.add(Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform',
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))

    model.summary()

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=SGD(lr=params['lr'], momentum=0.9, nesterov=True))

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, min_lr=0.00001, verbose=1)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=10000,
              callbacks=[reduce_lr, keras.callbacks.EarlyStopping(min_delta=1e-2, patience=20)])

    print(model.summary())

    y_pred = model.predict_classes(x_predict)
    metric = matthews_corrcoef(y_predict, y_pred)

    #    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    return -metric

start_clf = time.time()

params_space_DNN = {'dropout': hp.uniform('dropout', 0.0, 0.5),
                    'lr': hp.loguniform('lr', np.log(0.0001), np.log(1)),
                   'activation_type': hp.choice('activation_type', ['tanh', 'relu']),
                   'n_layers': hp.choice('n_layers', range(2, 6))}

trials = hyperopt.Trials()


best = hyperopt.fmin(hyperopt_DNN_objective,
                     space=params_space_DNN,
                     algo=tpe.suggest,
                     max_evals=30,
                     trials=trials,
                     rstate=RandomState(0))

tr_time_clf = (time.time() - start_clf)/60
print('Training time is %.2f min' % tr_time_clf)

print('best:')
print(best)

parameters = ['dropout', 'lr', 'activation_type', 'n_layers']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(25, 5))
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
    axes[i].set_title("DNN")
    axes[i].set_ylim([0.0, 1.0])
    axes[i].set_ylabel("MCC", fontsize=12)
    axes[i].set_xlabel(val, fontsize=12)

plt.show()