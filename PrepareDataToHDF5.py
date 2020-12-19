import numpy as np
import pandas as pd
import csv
import h5py
import os
from sklearn import model_selection
from sklearn.model_selection import KFold

"""
This script can be utilized to split a dataset into a train/validation set and subsequently split the train file into 5-fold cross validation.
Requires the python libraries http://www.h5py.org/ and http://scikit-learn.org/stable/ to be installed on the system.
"""

def read_csv(ifile):
	"""Reads csv into numpy arrays:
	csv has to be in format: CID,SMILE,Activity label,features"""
	print("Now working on file{}".format(ifile))
	reader = csv.reader(open(ifile, "rt"), delimiter=',')
	headers = next(reader)
	reader = pd.read_csv(ifile)
	labels = reader['Activity']
	features = reader.iloc[:, 3:]
	labels = np.array(labels, dtype=float)
	features = np.array(features, dtype=float)
	print("Number of instances: ", len(labels))
	print("Number of features", len(features[0]))
	print()
	print("Done reading csv file")
	print()
	return labels, features


def TrainTest_Validation(labels,features):
	"Splits train-test and validation dataset: 80% is used for training-testing (5Fold Cross Validation) and 20% is kept outside for model validation"
	print("Now splitting to test and validation sets")
	train_labels, test_labels, train_features, test_features = model_selection.train_test_split(labels, features, test_size=0.20, random_state=0)
	print(len(train_labels))
	print(len(test_labels))
	print("Done")
	return train_labels, test_labels, train_features, test_features


def Kfold(train_labels, train_features):
	"Splits training dataset in 5 fold cross validation pair sets train1/test1, train2/test2 etc"
	sss = KFold(n_splits=5, shuffle=True, random_state=0)
	print(len(train_labels))
	return sss


def write_TrainSet(fileName, train_labels, train_features):
	"""
	Writes the complete training dataset to file
	"""
	print(len(train_labels))
	os.chdir("E:/My Coding/Scripts/Path to/Train set/lipinski_train_test/")
	HDF5file = fileName+".h5"
	CSVfile = fileName+".csv"
	print("Now writing train set")
	print(HDF5file)
	print(CSVfile)
	writerTrain = csv.writer(open(CSVfile, "wt", newline=''), delimiter=',')
	with h5py.File(HDF5file, 'w') as f:
		f['HDF5Data2'] = train_labels.astype(np.float32)
		f['HDF5Data1'] = train_features.astype(np.float32)

	for j in range(len(train_labels)):
		temp=[]
		temp.append(int(train_labels[j]))
		for x in range(len(train_features[j])):
			temp.append(float(train_features[j][x]))

		writerTrain.writerow(temp)
	return



def write_Validation(fileName, test_labels, test_features):
	"""
	Writes the validation set to file
	"""
	print(len(test_labels))
	os.chdir("E:/My Coding/Scripts/Path to/Validation/lipinski/")
	HDF5file = fileName+".h5"
	CSVfile = fileName+".csv"
	print("Now writing test set")
	print(HDF5file)
	print(CSVfile)
	writerValidation = csv.writer(open(CSVfile, "wt", newline=''), delimiter=',')
	with h5py.File(HDF5file, 'w') as f:
		f['HDF5Data1'] = test_features.astype(np.float32)
		f['HDF5Data2'] = test_labels.astype(np.float32)

	for j in range(len(test_labels)):
		temp = []
		temp.append(int(test_labels[j]))
		for x in range(len(test_features[j])):
			temp.append(float(test_features[j][x]))

		writerValidation.writerow(temp)
	return


def write_files(sss, labels, features):
	os.chdir("E:/My Coding/Scripts/Path to/Train_Test/lipinski/")
	count = 0
	for train_index, test_index in sss.split(train_labels):
		print()
		count += 1
		print("Fold: ", count)
		#==================
		#Writes outputfiles to csv files
		Train = "train_"+str(count)
		CsvTrain = Train+".csv"
		H5Train = Train+".h5"
		Test = "test_"+str(count)
		CsvTest = Test+".csv"
		H5Test = Test+".h5"
		y_train = labels[train_index] #train labels
		y_test = labels[test_index]   #test labels
		x_train = features[train_index] #train features
		x_test = features[test_index]  #test features
		writerTrain = csv.writer(open(CsvTrain, "wt", newline=''), delimiter=',')
		writerTest = csv.writer(open(CsvTest, "wt", newline=''), delimiter=',')
		print("Number of training instances ", len(x_train))
		print("Number of test instances ", len(x_test))
		print()
		print("Now writing to ", CsvTrain, H5Train)
		with h5py.File(H5Train, 'w') as f:
			f['HDF5Data1'] = x_train.astype(np.float32)
			f['HDF5Data2'] = y_train.astype(np.float32)
		with h5py.File(H5Test, 'w') as f:
			f['HDF5Data1'] = x_test.astype(np.float32)
			f['HDF5Data2'] = y_test.astype(np.float32)
		for j in range(len(y_train)):

			temp = []
			temp.append(int(y_train[j]))
			for x in range(len(x_train[j])):
				temp.append(float(x_train[j][x]))

			writerTrain.writerow(temp)

		print()
		print("Now writing to ", CsvTest)

		for j in range(len(y_test)):
			temp = []
			temp.append(int(y_test[j]))
			for x in range(len(x_test[j])):
				temp.append(float(x_test[j][x]))

			writerTest.writerow(temp)


		#======================
		#Writes outputfiles to hd5 files
		Train="train_"+str(count)
		CsvTrain=Train+".csv"
		Test="test_"+str(count)
		CsvTest=Test+".csv"
	return



if __name__ == '__main__':
	labels, features = read_csv("E:/My Coding/Scripts/Path to/Train set/lipinski_ecfp/lipinski_ecfp6_1024.csv")
	train_labels, test_labels, train_features, test_features = TrainTest_Validation(labels, features)
	write_Validation("validation", test_labels, test_features)
	write_TrainSet("train_test", train_labels, train_features)
	sss = Kfold(train_labels, train_features)
	write_files(sss, train_labels, train_features)
