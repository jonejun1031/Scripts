from sklearn.naive_bayes import BernoulliNB
import csv
import numpy as np
from sklearn.metrics import auc, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
from sklearn.metrics import accuracy_score


def BernoulNB(X_train, Y_train, x_test, y_test, a, binz, cl_pr, ft_pr):
    nb = BernoulliNB(alpha=float(a), binarize=float(binz), class_prior=cl_pr, fit_prior=ft_pr)
    nb.fit(X_train, Y_train)

    y_train_pred_prob = nb.predict_proba(X_train)
    y_train_pred = nb.predict(X_train)
#    print(y_train_pred_prob)
    temp_train = []
#    print(y_train_pred_prob)
    for j in range(len(y_train_pred_prob)):
        temp_train.append(y_train_pred_prob[j][1])

    y_pred_prob = nb.predict_proba(x_test)
    y_pred = nb.predict(x_test)
#    print(y_pred_prob)
    temp = []
#    print(y_pred_prob)
    for j in range(len(y_pred_prob)):
        temp.append(y_pred_prob[j][1])

    auc = roc_auc_score(np.array(y_test), np.array(temp))
    auc_train = roc_auc_score(np.array(Y_train), np.array(temp_train))
    acc = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(Y_train, y_train_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    mcc_train = matthews_corrcoef(Y_train, y_train_pred)
    CK = cohen_kappa_score(y_test, y_pred)
    CK_train = cohen_kappa_score(Y_train, y_train_pred)
    Recall = recall_score(y_test, y_pred, pos_label=1)
    Recall_train = recall_score(Y_train, y_train_pred, pos_label=1)
    Precision = precision_score(y_test, y_pred, pos_label=1)
    Precision_train = precision_score(Y_train, y_train_pred, pos_label=1)
    F1_score = f1_score(y_test, y_pred, pos_label=1)
    F1_score_train = f1_score(Y_train, y_train_pred, pos_label=1)
    print(auc, acc, mcc, CK, Recall, Precision, F1_score, auc_train,
          acc_train, mcc_train, CK_train, Recall_train, Precision_train, F1_score_train)
    return y_test, y_pred_prob, y_pred, auc, acc, mcc, CK, Recall, Precision, F1_score, auc_train, acc_train, mcc_train,\
           CK_train, Recall_train, Precision_train, F1_score_train

def BNB_predict(ifile1, ifile2, ofile1, ofile2, a, binz, cl_pr, ft_pr):
    print("Now reading train file: ", ifile1)
    Xreader = csv.reader(open(ifile1, "rt"), delimiter=',')
    X_Train = []
    Y_Train = []
    countTrain = 0
    for row in Xreader:
        X_Train.append(np.array(row[1:], dtype=float))
        Y_Train.append(int(row[0]))
        countTrain += 1

    print()
    print("Now reading test file: ", ifile2)

    xreader = csv.reader(open(ifile2, "rt"), delimiter=',')

    x_test = []
    y_test = []
    countTest = 0
    for row in xreader:
        countTest += 1
        x_test.append(np.array(row[1:], dtype=float))
        y_test.append(int(row[0]))

    print()
    print("Training Samples, ", countTrain)
    print("Test Samples", countTest)
#    print(len(x_test))
#    print(len(y_test))

    y_test, y_pred_prob, y_pred, auc, acc, mcc, CK, Recall, Precision, F1_score, auc_train, acc_train, mcc_train, \
    CK_train, Recall_train, Precision_train, F1_score_train = BernoulNB(X_Train, Y_Train, x_test, y_test, a, binz, cl_pr, ft_pr)

    writer = csv.writer(open(ofile1, "wt", newline=''), delimiter=',')
    writer2 = csv.writer(open(ofile2, "wt", newline=''), delimiter=',')


    newHeaders = ['auc', 'acc', 'mcc', 'CK', 'Recall', 'Precision', 'F1_score', 'auc_train', 'acc_train', 'mcc_train',
                  'CK_train', 'Recall_train', 'Precision_train', 'F1_score_train']
#    print(y_test)
    temp = []
    for j in range(len(y_pred_prob)):
        temp.append(y_pred_prob[j][1])
    auc = roc_auc_score(np.array(y_test), np.array(temp))
    writer2.writerow(newHeaders)
    writer2.writerow([auc, acc, mcc, CK, Recall, Precision, F1_score, auc_train, acc_train,
                      mcc_train, CK_train, Recall_train, Precision_train, F1_score_train])
    print("AUC", auc)
    headers = ['Y_true', 'Y_predict_NB']
    writer.writerow(headers)
    for j in range(len(y_test)):
        writer.writerow([y_test[j], temp[j]])
    return

if __name__ == '__main__':
    BNB_predict("E:/My Coding/Scripts/Path to/Train set/lipinski_train_test/train_test.csv",
                "E:/My Coding/Scripts/Path to/Validation/lipinski/validation.csv",
                "E:/My Coding/Scripts/Path to output predictions/lipinski/NB_Validation_Predictions.csv",
                "E:/My Coding/Scripts/Path ro output metrics/lipinski/NB_Validation_Metrics.csv", 1.0258767358679375, 0, None, True)

