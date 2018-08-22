from sklearn import linear_model
import numpy as np;
def learnBayes(data, labels):
	clf = linear_model.BayesianRidge();
	print len(data), len(data[0]), len(labels)
	clf.fit(data, labels);
	return clf;
