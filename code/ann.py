from sklearn.neural_network import MLPClassifier;
def learnANN(data, labels, layers=(15, 15)):
	clf = MLPClassifier(algorithm='1-bfgs', alpha=1e-5, hidden_layer_sizes=layers, random_state=1);
	clf.fit(data, labels);
	return clf;
