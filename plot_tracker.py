import pickle
import pybertini
import math
import matplotlib.pyplot as plt
import numpy as np


path_information = pickle.load(open("data_set.p", "rb"))

# scatter plot
#use diff color for each predictor
#log tol (xaxis) vs log time (yaxis)

fig, ax = plt.subplots()
trackers = np.zeros(shape=(path_information.shape[0], 1))

for ii in range(path_information.shape[0]):
	thing = path_information[ii]
	trackers[ii] = thing[1]
	row = path_information[ii]
	x = math.log10(row[0]) * -1
	y = math.log10(row[2]) * -1

	if ii < 7:
		if trackers[ii] == [1.]:
			plt.scatter(x, y, c = 'red', label = 'Euler')
		elif trackers[ii] == [4.]:
			plt.scatter(x, y, c = 'blue', label = 'HeunEuler')
		elif trackers[ii] == [3.]:
			plt.scatter(x, y, c = 'green', label = 'RK4')
		elif trackers[ii] == [7.]:
			plt.scatter(x, y, c = 'cyan', label = 'RKCashKarp45')
		elif trackers[ii] == [8.]:
			plt.scatter(x, y, c = 'magenta', label = 'RKDormandPrince56')
		elif trackers[ii] == [6.]:
			plt.scatter(x, y, c = 'yellow', label = 'RKF45')
		elif trackers[ii] == [9.]:
			plt.scatter(x, y, c = 'black', label = 'RKVerner67')
	else:
		if trackers[ii] == [1.]:
			plt.scatter(x, y, c = 'red')
		elif trackers[ii] == [4.]:
			plt.scatter(x, y, c = 'blue')
		elif trackers[ii] == [3.]:
			plt.scatter(x, y, c = 'green')
		elif trackers[ii] == [7.]:
			plt.scatter(x, y, c = 'cyan')
		elif trackers[ii] == [8.]:
			plt.scatter(x, y, c = 'magenta')
		elif trackers[ii] == [6.]:
			plt.scatter(x, y, c = 'yellow')
		elif trackers[ii] == [9.]:
			plt.scatter(x, y, c = 'black')



plt.legend(loc = 'upper right')
fig.suptitle("Plot of Tracking Paths")
plt.xlabel('log(tracking tolerance)')
plt.ylabel('log(time to execute)')
plt.show()
#fig.savefig('test.png')