import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def GetDataByPandas(ifile):
    data = pd.read_csv(ifile)
    y = np.array(data['class'])
    X = np.array(data.drop("class", axis=1))
    columns = np.array(data.columns)

    return X, y, columns

def DNN(x_train, y_train, x_test, y_test, x_predict, y_predict, activation_type, dropout, lr, n_layers):
    model = tf.keras.models.Sequential()
    model.add(Dense(1024, input_dim=1024, kernel_initializer='lecun_uniform',
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
    model.add(Activation(activation_type))
    model.add(Dropout(float(dropout)))

    for idx in range(int(n_layers)-1):
        model.add(Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=0.005),
                        kernel_initializer='lecun_uniform'))
        model.add(Activation(activation_type))
        model.add(Dropout(float(dropout)))

    model.add(Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform',
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
    model.summary()

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=SGD(lr=lr, momentum=0.9, nesterov=True))

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, min_lr=0.00001, verbose=1)

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=64, epochs=10000,
              callbacks=[reduce_lr, keras.callbacks.EarlyStopping(min_delta=1e-2, patience=20)])

    print(model.summary())

    y_train_pred_prob = model.predict_proba(x_train)
    y_train_pred = model.predict_classes(x_train)
    temp_train = []

    for j in range(len(y_train_pred_prob)):
        temp_train.append(y_train_pred_prob[j])

    y_pred_prob = model.predict_proba(x_predict)
    y_pred = model.predict_classes(x_predict)
    temp = []

    for j in range(len(y_pred_prob)):
        temp.append(y_pred_prob[j])

    auc = roc_auc_score(np.array(y_predict), np.array(temp))
    auc_train = roc_auc_score(np.array(y_train), np.array(temp_train))
    acc = accuracy_score(y_predict, y_pred)
    acc_train = accuracy_score(y_train, y_train_pred)
    mcc = matthews_corrcoef(y_predict, y_pred)
    mcc_train = matthews_corrcoef(y_train, y_train_pred)
    CK = cohen_kappa_score(y_predict, y_pred)
    CK_train = cohen_kappa_score(y_train, y_train_pred)
    Recall = recall_score(y_predict, y_pred, pos_label=1)
    Recall_train = recall_score(y_train, y_train_pred, pos_label=1)
    Precision = precision_score(y_predict, y_pred, pos_label=1)
    Precision_train = precision_score(y_train, y_train_pred, pos_label=1)
    F1_score = f1_score(y_predict, y_pred, pos_label=1)
    F1_score_train = f1_score(y_train, y_train_pred, pos_label=1)
    print(auc, acc, mcc, CK, Recall, Precision, F1_score, auc_train,
          acc_train, mcc_train, CK_train, Recall_train, Precision_train, F1_score_train)
    return y_test, y_pred_prob, y_pred, auc, acc, mcc, CK, Recall, Precision, F1_score, \
           auc_train, acc_train, mcc_train, CK_train, Recall_train, Precision_train, F1_score_train

def DNN_predict(ifile, ofile1, ofile2, activation_type, dropout, lr, n_layers):
    X, y, Names = GetDataByPandas(ifile)

    # split data to [[0.8,0.2],01]
    x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.20, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=0)

    y_test, y_pred_prob, y_pred, auc, acc, mcc, CK, Recall, Precision, F1_score, auc_train, acc_train, mcc_train, \
    CK_train, Recall_train, Precision_train, F1_score_train = DNN(x_train, y_train, x_test, y_test, x_predict, y_predict,
                                                                  activation_type, dropout, lr, n_layers)

    writer = csv.writer(open(ofile1, "wt", newline=''), delimiter=',')
    writer2 = csv.writer(open(ofile2, "wt", newline=''), delimiter=',')


    newHeaders = ['auc', 'acc', 'mcc', 'CK', 'Recall', 'Precision', 'F1_score', 'auc_train', 'acc_train', 'mcc_train',
                  'CK_train', 'Recall_train', 'Precision_train', 'F1_score_train']
    temp = []
    for j in range(len(y_pred_prob)):
        temp.append(y_pred_prob[j])
    auc = roc_auc_score(np.array(y_predict), np.array(temp))
    writer2.writerow(newHeaders)
    writer2.writerow([auc, acc, mcc, CK, Recall, Precision, F1_score, auc_train, acc_train,
                      mcc_train, CK_train, Recall_train, Precision_train, F1_score_train])
    print("AUC", auc)
    headers = ['Y_true', 'Y_predict_AdaBoost']
    writer.writerow(headers)
    for j in range(len(y_test)):
        writer.writerow([y_test[j], temp[j]])

    return

if __name__ == '__main__':
    DNN_predict("E:/My Coding/Scripts/Data for hyperopt/lipinski_ecfp6_1024.csv",
                "E:/My Coding/Scripts/Path to output predictions/lipinski/DNN_Validation_Predictions.csv",
                "E:/My Coding/Scripts/Path ro output metrics/lipinski/DNN_Validation_Metrics.csv",
                'relu', 0.111609750471036, 0.011877960679976696, 5)

