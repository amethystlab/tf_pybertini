import numpy as np
import pickle
import pybertini
import random
import sys
import time

def main(n):
	tols = 10 ** (np.linspace(-2, -10, n))
	import pybertini.tracking as tr
	P = tr.Predictor
	# preds = [p for p in pybertini.tracking.Predictor]
	# took the Heun and P.RKNorsett34 predictors out
	# omit Constant because it is slow
	preds = [P.Euler, P.HeunEuler, P.RK4, P.RKCashKarp45, P.RKDormandPrince56, P.RKF45, P.RKVerner67]
	data_set = create_data(tols, preds)
	pickle.dump(data_set, open('data_set.p','wb'))

def create_data(tracking_tolerances, predictors):

	array_of_data_sets = np.ndarray(len(tracking_tolerances) * len(predictors), dtype=np.object)
	i = 0
	for t in tracking_tolerances:
		for p in predictors:
			print('{} {}'.format(t, p))
			#str(t) use this if tk load fails
			runtime, successcode = compute_values(t, p)
			array_of_data_sets[i] = (t, p, runtime, successcode)
			i += 1

	return array_of_data_sets

def compute_values(tol, pred):
	gw = pybertini.System()

	x = pybertini.Variable("x")
	y = pybertini.Variable("y")

	vg = pybertini.VariableGroup()
	vg.append(x)
	vg.append(y)
	gw.add_variable_group(vg)

	gw.add_function(pybertini.function_tree.symbol.Rational('29/16')*x**3 - 2*x*y)
	gw.add_function(y - x**2)

	t = pybertini.Variable('t')
	td = pybertini.system.start_system.TotalDegree(gw)
	gamma = pybertini.function_tree.symbol.Rational.rand()
	hom = (1-t)*gw + t*gamma*td
	hom.add_path_variable(t)

	tr = pybertini.tracking.AMPTracker(hom)
	tr.tracking_tolerance(tol)
	tr.predictor(pred)

	start_time = pybertini.multiprec.Complex("1")
	eg_boundary = pybertini.multiprec.Complex("0.1")

	midpath_points = [None]*td.num_start_points()
	overall_code = pybertini.tracking.SuccessCode.Success
	speed_start = time.time()

	for ii in range(td.num_start_points()):
		print("Tracking path {}...".format(ii))
		midpath_points[ii] = pybertini.multiprec.Vector()
		code = tr.track_path(result=midpath_points[ii], start_time=start_time, end_time=eg_boundary, start_point=td.start_point_mp(ii))
		if code != pybertini.tracking.SuccessCode.Success:
			overall_code = pybertini.tracking.SuccessCode.Failure



	return time.time() - speed_start, overall_code

if __name__ == '__main__':
	main(3)