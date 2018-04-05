import pickle
import pybertini
import math
import matplotlib.pyplot as plt
import numpy as np


path_information = pickle.load(open("data_set.p", "rb"))
n = 3 # number of times run / number of points for each tracker
# scatter plot
#use diff color for each predictor
#log tol (xaxis) vs log time (yaxis)

fig, ax = plt.subplots()
trackers = np.zeros(shape=(path_information.shape[0], 1))
predictor1x = np.zeros(n)
predictor1y = np.zeros(n)
predictor2x = np.zeros(n)
predictor2y = np.zeros(n)
predictor3x = np.zeros(n)
predictor3y = np.zeros(n)
predictor4x = np.zeros(n)
predictor4y = np.zeros(n)
predictor5x = np.zeros(n)
predictor5y = np.zeros(n)
predictor6x = np.zeros(n)
predictor6y = np.zeros(n)
predictor7x = np.zeros(n)
predictor7y = np.zeros(n)

# compute the data
for ii in range(path_information.shape[0]):
	thing = path_information[ii]
	trackers[ii] = thing[1]
	row = path_information[ii]
	x = math.log10(row[0]) * -1
	y = math.log10(row[2]) * -1

	#if ii < 7:
	if trackers[ii] == [1.]:
		predictor1x = np.append(predictor1x, x)
		predictor1y = np.append(predictor1y, y)
		#plt.scatter(x, y, marker = 'D', c = 'red', label = 'Euler')
	elif trackers[ii] == [4.]:
		predictor2x = np.append(predictor2x, x)
		predictor2y = np.append(predictor2y, y)
	# 	#plt.scatter(x, y, c = 'blue', label = 'HeunEuler')
	elif trackers[ii] == [3.]:
		predictor3x = np.append(predictor3x, x)
		predictor3y = np.append(predictor3y, y)
	# 	#plt.scatter(x, y, c = 'green', label = 'RK4')
	elif trackers[ii] == [7.]:
		predictor4x = np.append(predictor4x, x)
		predictor4y = np.append(predictor4y, y)
	# 	#plt.scatter(x, y, c = 'cyan', label = 'RKCashKarp45')
	elif trackers[ii] == [8.]:
		predictor5x = np.append(predictor5x, x)
		predictor5y = np.append(predictor5y, y)
	# 	#plt.scatter(x, y, c = 'magenta', label = 'RKDormandPrince56')
	elif trackers[ii] == [6.]:
		predictor6x = np.append(predictor6x, x)
		predictor6y = np.append(predictor6y, y)
	# 	#plt.scatter(x, y, c = 'yellow', label = 'RKF45')
	elif trackers[ii] == [9.]:
		predictor7x = np.append(predictor7x, x)
		predictor7y = np.append(predictor7y, y)
		#plt.scatter(x, y, c = 'black', label = 'RKVerner67')


#plot the data, predictor by predictor
	# print(predictor1x)
	#print(predictor1y)

plt.plot(predictor1x, predictor1y, linestyle = '--', linewidth = 3, color = 'red', label = 'Euler')
plt.plot(predictor2x, predictor2y, linestyle = '--', linewidth = 3, color = 'blue', label = 'HeunEuler')
plt.plot(predictor3x, predictor3y, linestyle = '--', linewidth = 3, color = 'green', label = 'RK4')
plt.plot(predictor4x, predictor4y, linestyle = '--', linewidth = 3, color = 'cyan', label = 'RKCashKarp45')
plt.plot(predictor5x, predictor5y, linestyle = '--', linewidth = 3, color = 'magenta', label = 'RKDormandPrince56')
plt.plot(predictor6x, predictor6y, linestyle = '--', linewidth = 3, color = 'yellow', label = 'RKF45')
plt.plot(predictor7x, predictor7y, linestyle = '--', linewidth = 3, color = 'black', label = 'RKVerner67')



plt.legend(loc = 'upper right')
fig.suptitle("Plot of Tracking Paths")
plt.xlabel('log(tracking tolerance)')
plt.ylabel('log(time to execute)')
plt.style.use("seaborn-darkgrid")
plt.show()
#fig.savefig('test.png')