import csv
import numpy as np
from scipy import stats
import helpers
from sklearn.decomposition import PCA

class patient():
	def __init__(self, id, age, label):
		self.id = id
		self.age = age
		self.label = label
		self.time = []
		self.icu = []
		self.labs = [[] for i in range(25)]
		self.vitals = [[] for i in range(6)]
		self.cleanlabs = [[] for i in range(25)]
		self.cleanvitals = [[] for i in range(6)]

	def new_stats(self, row_labs, row_vitals):
		self.time.append(int(row_labs[1]))
		self.icu.append(row_vitals[8])
		for i in range(2,27):
			if(row_labs[i]) != 'NA':
				row_labs[i] = float(row_labs[i])
			self.labs[i-2].append(row_labs[i])

		for i in range(2,8):
			if(row_vitals[i]) != 'NA':
				row_vitals[i] = float(row_vitals[i])
			self.vitals[i-2].append(row_vitals[i])

	def cleanup(self):
		for i in range(25):
			for j in range(len(self.labs[i])):
				if(self.labs[i][j]) != 'NA':
					self.cleanlabs[i].append(self.labs[i][j])

		for i in range(6):
			for j in range(len(self.vitals[i])):
				if(self.vitals[i][j]) != 'NA':
					self.cleanvitals[i].append(self.vitals[i][j])


def csv_reader(file_path):
	"""
	Return a csv file reader
	"""
	file_obj = open(file_path, "r")
	reader = csv.reader(file_obj)
	return file_obj,reader

def close_csv_reader(file_obj):
	file_obj.close()

def get_next(file_obj):
	return file_obj.next()

def extract_data(dataset_prefix = "../dataset/Training_Dataset/"):
	age_file, age_reader = csv_reader(dataset_prefix + "id_age_train.csv")
	label_file, label_reader = csv_reader(dataset_prefix + "id_label_train.csv")
	labs_file, labs_reader = csv_reader(dataset_prefix + "id_time_labs_train.csv")
	vitals_file, vitals_reader = csv_reader(dataset_prefix + "id_time_vitals_train.csv")

	get_next(age_reader)
	get_next(label_reader)
	get_next(labs_reader)
	get_next(vitals_reader)


	cur_id = -1

	temp = ""

	training_data = []

	temp = ""
	i = 0
	while 1:
		i = i+1
		try:
			labs_next = get_next(labs_reader)
			vitals_next = get_next(vitals_reader)

			if(labs_next[0] != cur_id):
				if(temp != ""):
					training_data.append(temp)

				age_next = get_next(age_reader)
				label_next = get_next(label_reader)

				cur_id = age_next[0]
				temp = patient(age_next[0], age_next[1], label_next[1])
				temp.new_stats(labs_next, vitals_next)

			else:
				temp.new_stats(labs_next, vitals_next)
		except:
			print i,"yay"
			training_data.append(temp)
			break;


	close_csv_reader(age_file)
	close_csv_reader(label_file)
	close_csv_reader(labs_file)
	close_csv_reader(vitals_file)

	for i in training_data:
		i.cleanup()

	formatted_data = []
	labels = []
	for i in training_data:
		data = [int(i.age)]
		labels.append(int(i.label))
		for ind, j in enumerate(i.cleanlabs):
			deri = helpers.derivative(i.time, i.labs[ind])
			quartiles = helpers.quartile(j)
			data.append(helpers.one_if_all_non_zero(j))
			data.append(helpers.first(j) - helpers.last(j))
			data.append(helpers.first(j))
			#data.append(0)	#KURTOSIS
			data.append(helpers.Max(deri))
			data.append(helpers.Max(deri) - helpers.Min(deri))
			data.append(helpers.Max(j))
			data.append(helpers.mean(deri))
			data.append(helpers.mean(j))
			data.append(abs(helpers.mean(j) - helpers.median(j)))
			data.append(helpers.median(deri))
			data.append(helpers.median(j))
			data.append(helpers.Min(j))
			data.append(helpers.mode(j))
			data.append(len(j))
			data.append(quartiles[0])
			data.append(quartiles[1])
			data.append(helpers.Max(j) - helpers.Min(j))
			data.append(helpers.signum(helpers.mean(deri)))
			data.append(helpers.std(deri))
			data.append(helpers.std(j))
			data.append(sum(j))
			data.append(helpers.std(j)**2)
			data.append(helpers.std(deri)**2)
		for ind, j in enumerate(i.cleanvitals):
			deri = helpers.derivative(i.time, i.vitals[ind])
			quartiles = helpers.quartile(j)
			data.append(helpers.one_if_all_non_zero(j))
			data.append(helpers.first(j) - helpers.last(j))
			data.append(helpers.first(j))
			#data.append(0)	#KURTOSIS
			data.append(helpers.Max(deri))
			data.append(helpers.Max(deri) - helpers.Min(deri))
			data.append(helpers.Max(j))
			data.append(helpers.mean(deri))
			data.append(helpers.mean(j))
			data.append(abs(helpers.mean(j) - helpers.median(j)))
			data.append(helpers.median(deri))
			data.append(helpers.median(j))
			data.append(helpers.Min(j))
			data.append(helpers.mode(j))
			data.append(len(j))
			data.append(quartiles[0])
			data.append(quartiles[1])
			data.append(helpers.Max(j) - helpers.Min(j))
			data.append(helpers.signum(helpers.mean(deri)))
			data.append(helpers.std(deri))
			data.append(helpers.std(j))
			data.append(sum(j))
			data.append(helpers.std(j)**2)
			data.append(helpers.std(deri)**2)
		formatted_data.append(data)

	return (formatted_data, labels)


def getData(train, test):
	training_data = extract_data(train)
	testing_data = extract_data(test)
	l1 = len(training_data[0])
	formatted_data = training_data[0]
	formatted_data.extend(testing_data[0])

	zc = [0]*len(formatted_data[0])
	for i in formatted_data:
		for j in xrange(len(zc)):
			if i[j] == 0:
				zc[j] += 1

	clean_data = []
	for i in formatted_data:
		data = []
		for j in xrange(len(zc)):
			if zc[j] < 400:
				data.append(i[j])
		clean_data.append(data)

	print len(clean_data), len(clean_data[0])

	maxes = [-99999999999999]*len(clean_data[0]);

	for i in clean_data:
		for j in xrange(len(i)):
			if i[j] > maxes[j]:
				maxes[j] = i[j];

	for i in xrange(len(clean_data)):
		for j in xrange(len(clean_data[i])):
			clean_data[i][j] /= maxes[j];

	pca = PCA(20)
	pca.fit(np.array(clean_data).transpose())
	components = pca.components_.transpose()
	return ((components[:l1], training_data[1]), (components[l1:], testing_data[1]))
