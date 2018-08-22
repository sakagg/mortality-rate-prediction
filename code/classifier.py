#from ann import learnANN;
from bayes import learnBayes;
from svm import learnSVM;
from parse import getData;
import sys;
def classify(train, test):
	trainData, testData = getData(train, test);
	#print len(trainData[0]), len(testData[0])
	#print len(trainData[0][0]), len(trainData[1]), len(testData[0][0]), len(testData[1])
	#annClass = learnANN(trainData[0], trainData[1]);
	#annRes = annClass.predict(testData[0]);
	#annAcc = 0;
	bayesAcc = 0;
	svmAcc = 0;
	bayesClass = learnBayes(trainData[0], trainData[1]);
	bayesRes = bayesClass.predict(testData[0]);
	bayesRes = map(lambda x: 0 if x < 0.5 else 1, bayesRes)
	svmClass = learnSVM(trainData[0], trainData[1]);
	svmRes = svmClass.predict(testData[0]);
	svmRes = map(lambda x: 0 if x < 0.5 else 1, svmRes)
	for i in xrange(len(testData[1])):
		#if annRes[i] == testData[1][i]:
		#	annAcc += 1
		if bayesRes[i] == testData[1][i]:
			bayesAcc += 1
		if svmRes[i] == testData[1][i]:
			svmAcc += 1
	#print "ANN Accuracy:", annAcc/(len(testData[1])*1.0);
	print "Bayes Accuracy:", bayesAcc/(len(testData[1])*1.0);
	print "SVM Accuracy:", svmAcc/(len(testData[1])*1.0);

def main():
	classify(sys.argv[1], sys.argv[2]);

if __name__ == '__main__':
	main()
