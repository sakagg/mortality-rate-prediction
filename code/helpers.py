import numpy as np
from scipy import stats

def one_if_all_non_zero(arr):
	for i in arr:
		if i == 0:
			return 0
	return 1

def Max(arr):
	if len(arr) == 0:
		return 0
	return max(arr)

def Min(arr):
	if len(arr) == 0:
		return 0
	return min(arr)

def first(arr):
	if len(arr) == 0:
		return 0
	return arr[0]

def last(arr):
	if len(arr) == 0:
		return 0
	return arr[-1]

def std(arr):
	if len(arr) == 0:
		return 0
	return np.std(arr)

def signum(num):
	if num >0:
		return 1
	elif num <0:
		return -1
	return 0

def mean(array):
	if len(array) == 0:
		return 0
	return np.average(array)

def median(array):
	if len(array) == 0:
		return 0
	return np.median(array)

def mode(array):
	if len(array) == 0:
		return 0
	return stats.mode(array).mode[0]

def quartile(array):
	arr1 = []
	arr2 = []

	length = len(array)
	arrx = []
	for i in range(length):
		arrx.append(array[i])
	arrx.sort()
	if((len(array)%2) == 1):
		arr1 = array[:length/2]
		arr2 = array[length/2+1:]
	else:
		arr1 = array[:length/2]
		arr2 = array[length/2:]

	return (median(arr1), median(arr2))

def derivative(array_time, array_func):
	prev = ''
	prev_time = ''
	length = len(array_time)
	der = []

	for i in range(length):
		if(array_func[i] != 'NA'):
			if(prev != ''):
				der.append((array_func[i]-prev)/(array_time[i]-prev_time))
			prev = array_func[i]
			prev_time = array_time[i]

	return der
