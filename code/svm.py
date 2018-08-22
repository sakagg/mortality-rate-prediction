from sklearn import svm;
def learnSVM(data, labels):
	clf = svm.SVC();
	clf.fit(data, labels);
	return clf;
