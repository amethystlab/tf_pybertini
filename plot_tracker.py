import pickle
import pybertini
import math
import matplotlib.pyplot as plt
import numpy as np

path_information = pickle.load(open("data_set_10.p", "rb"))
n = 0
# scatter plot
#use diff color for each predictor
#log tol (xaxis) vs log time (yaxis)

print_logs_times = True

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

fails_x = np.zeros(n)
fails_y = np.zeros(n)

# compute the data
for ii in range(path_information.shape[0]):
	thing = path_information[ii]
	trackers[ii] = thing[1]
	row = path_information[ii]

	# x = tolerance
	x = math.log10(row[0]) * -1
	# y = track time
	if (print_logs_times):
		y = math.log10(row[2])
	else:
		y = row[2]

	# true if success
	z = row[3] == 0

	if not z:
		fails_x = np.append(fails_x,x)
		fails_y = np.append(fails_y,y)

	if trackers[ii] == [1.]:
		predictor1x = np.append(predictor1x, x)
		predictor1y = np.append(predictor1y, y)

	elif trackers[ii] == [4.]:
		predictor2x = np.append(predictor2x, x)
		predictor2y = np.append(predictor2y, y)

	elif trackers[ii] == [3.]:
		predictor3x = np.append(predictor3x, x)
		predictor3y = np.append(predictor3y, y)

	elif trackers[ii] == [7.]:
		predictor4x = np.append(predictor4x, x)
		predictor4y = np.append(predictor4y, y)

	elif trackers[ii] == [8.]:
		predictor5x = np.append(predictor5x, x)
		predictor5y = np.append(predictor5y, y)

	elif trackers[ii] == [6.]:
		predictor6x = np.append(predictor6x, x)
		predictor6y = np.append(predictor6y, y)

	elif trackers[ii] == [9.]:
		predictor7x = np.append(predictor7x, x)
		predictor7y = np.append(predictor7y, y)


#plot the data, predictor by predictor

plt.plot(predictor1x, predictor1y, linestyle = '-', linewidth = 2, color =  '#FF007F', label = 'Euler')
plt.plot(predictor2x, predictor2y, linestyle = '--', linewidth = 2, color = '#F49609', label = 'HeunEuler')
plt.plot(predictor3x, predictor3y, linestyle = '-.', linewidth = 2, color = '#84C318', label = 'RK4')
plt.plot(predictor4x, predictor4y, linestyle = ':', linewidth = 2, color = '#301EED', label = 'RKCashKarp45')
plt.plot(predictor5x, predictor5y, linestyle = '--', dashes = (5, 2, 20, 2), linewidth = 2, color = '#08A045', label = 'RKDormandPrince56')
plt.plot(predictor6x, predictor6y, linestyle = '--', dashes = (2, 5), linewidth = 2, color = '#7F0799', label = 'RKF45')
plt.plot(predictor7x, predictor7y, linestyle = '--', dashes = (5, 2), linewidth = 2, color = 'blue', label = 'RKVerner67')
plt.scatter(fails_x, fails_y, marker = 'o', color = 'red', label = 'Failure occured')


plt.legend(loc = 'best', prop={'size': 'x-small'})
fig.suptitle("Predictor and Tracking Tolerance Performance")
plt.xlabel('-log(tracking tolerance)')
if (print_logs_times):
	plt.ylabel('log(time to execute)')
else:
	plt.ylabel('time to execute')
plt.show()
#fig.savefig('test.png')
