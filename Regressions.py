from sklearn.cross_validation import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from math import sqrt
from copy import deepcopy
import numpy as np
import pandas as pd
from AutomobileRead import AutomobileRead
import matplotlib.pyplot as plt

kernel = 1 * RBF(1, (1e-3, 1e3))
def loop_byrec(k, finalK, params, regr, cv_data, res, partial=[]):
	if len(partial) == finalK:
		if regr=='Ridge':
			predictor = Ridge(alpha=partial[0])
		elif regr=='KernelRidgeLinear':
			predictor = KernelRidge(alpha = partial[0])
		elif regr=='KernelRidgeRBF':
			predictor = KernelRidge(alpha = partial[0], kernel='rbf', gamma=partial[1])
		elif regr=='GPR':
			global kernel
			kernel = 1 * RBF(partial[1], (1e-4, 1e4))
			predictor = GaussianProcessRegressor(alpha = partial[0], kernel=kernel)
		else:
			raise Exception()
		cv_train_data = cv_data[0]
		cv_train_y = cv_data[1]
		cv_test_data = cv_data[2]
		cv_test_y = cv_data[3]
		predictor.fit(cv_train_data, cv_train_y)
		predicted = predictor.predict(cv_test_data)
		res.append((sqrt(sum(pow(i-j, 2) for i, j in zip(predicted, cv_test_y))), deepcopy(partial)))
	else:
		for each_param in params[k]:
			partial.append(each_param)
			loop_byrec(k+1, finalK, params, regr, cv_data, res, partial)
			partial.pop()

class Regressions(object):
	def __init__(self, n_iteration=100, impute_type='mean'):
		ar = AutomobileRead()
		df = ar.GetImputedDataframe(ar.DoCategorizeFrom(ar.automobile), impute_type=impute_type)
		self.X = np.array(df.drop(['normalized-losses'], axis=1))
		self.Y = np.array(df['normalized-losses'])
		
		self.n_folds = 5
		self.n_iteration = n_iteration
		self.regrs = ['LinearRegression', 'Ridge', 'KernelRidgeLinear', 'KernelRidgeRBF', 'GPR']
		self.regrs_params = [\
			                [], \
			                [[pow(10, i) for i in range(-10, 11)]], \
			                [[pow(10, i) for i in range(-10, 11)]] , \
			                [[pow(10, i) for i in range(-30, -4)], [pow(10, i) for i in range(-30, -4)]], \
			                [[pow(10, i) for i in range(-5, 6)], [pow(10, i) for i in range(-5, 6)]] \
		               ]
		# self.regrs = ['LinearRegression', 'Ridge', 'KernelRidgeLinear']
		# self.regrs_params = [\
		# 	[], \
		# 	[[pow(10, i) for i in range(-10, 11)]], \
  #           [[pow(10, i) for i in range(-10, 11)]], \
		# ]
	def GetPredictor(self, regr, best_errs):
		if regr == 'Ridge':
			predictor = Ridge(alpha = best_errs[1][0])
		elif regr == 'LinearRegression':
			predictor = linear_model.LinearRegression()
		elif regr == 'KernelRidgeLinear':
			predictor = KernelRidge(alpha = best_errs[1][0])
		elif regr == 'KernelRidgeRBF':
			predictor = KernelRidge(alpha = best_errs[1][0], kernel='rbf', gamma = best_errs[1][1])
		elif regr == 'GPR':
			global kernel
			kernel = 1 * RBF(best_errs[1][1], (1e-4, 1e4))
			predictor = GaussianProcessRegressor(alpha = best_errs[1][0], kernel=kernel)
		else:
			raise Exception()
		return predictor
	def DoBatchPredict(self):
		result = []
		kf = KFold(n = len(self.X), n_folds=self.n_folds, shuffle=True)
		regrs_lenParam = [len(i) for i in self.regrs_params]
		for regr_ind, regr in enumerate(self.regrs):
			total_err = 0.
			for itera in range(self.n_iteration):
				for train_index, test_index in kf:
					train_data = self.X[train_index]
					test_data = self.X[test_index]
					train_y = self.Y[train_index]
					test_y = self.Y[test_index]
					kf2 = KFold(n = len(train_data), n_folds=self.n_folds, shuffle=True)
					best_errs = [float('inf'), []]
					if regrs_lenParam[regr_ind] != 0:
						for cv_train_ind, cv_test_ind in kf2:
							cv_train_data = train_data[cv_train_ind]
							cv_train_y = train_y[cv_train_ind]
							cv_test_data = train_data[cv_test_ind]
							cv_test_y = train_y[cv_test_ind]
							errs = []
							loop_byrec(0, regrs_lenParam[regr_ind], self.regrs_params[regr_ind], regr, \
								[cv_train_data, cv_train_y, cv_test_data, cv_test_y], errs)
							sorted_errs = sorted(errs, key=lambda x:x[0])
							if sorted_errs[0][0] < best_errs[0]:
								best_errs = sorted_errs[0]
					predictor = self.GetPredictor(regr, best_errs)
					predictor.fit(train_data, train_y)
					predicted = predictor.predict(test_data)
					total_err += sqrt(sum(pow(i-j, 2) for i, j in zip(predicted, test_y)))
			each_res = (regr, total_err)
			result.append(each_res)
			print(each_res)
		return result
	def CheckPlot(self, regr='LinearRegression'):
		fig = plt.figure(figsize=(18, 5))
		kf = KFold(n = len(self.X), n_folds=self.n_folds, shuffle=True)
		regrs_lenParam = [len(i) for i in self.regrs_params]
		try:
			regr_ind = self.regrs.index(regr)
		except:
			raise Exception()
		total_err = 0.
		print(regr)
		for fold_ind, (train_index, test_index) in enumerate(kf):
			herr = 0.
			train_data = self.X[train_index]
			test_data = self.X[test_index]
			train_y = self.Y[train_index]
			test_y = self.Y[test_index]
			kf2 = KFold(n = len(train_data), n_folds=self.n_folds, shuffle=True)
			best_errs = [float('inf'), []]
			if regrs_lenParam[regr_ind] != 0:
				for cv_train_ind, cv_test_ind in kf2:
					cv_train_data = train_data[cv_train_ind]
					cv_train_y = train_y[cv_train_ind]
					cv_test_data = train_data[cv_test_ind]
					cv_test_y = train_y[cv_test_ind]
					errs = []
					loop_byrec(0, regrs_lenParam[regr_ind], self.regrs_params[regr_ind], regr, \
						[cv_train_data, cv_train_y, cv_test_data, cv_test_y], errs)
					sorted_errs = sorted(errs, key=lambda x:x[0])
					if sorted_errs[0][0] < best_errs[0]:
						best_errs = sorted_errs[0]
			predictor = self.GetPredictor(regr, best_errs)
			predictor.fit(train_data, train_y)
			predicted = predictor.predict(test_data)
			herr = sqrt(sum(pow(i-j, 2) for i, j in zip(predicted, test_y)))
			total_err += herr
			print('CV Fold-{}, MSE: {}'.format(fold_ind+1, herr))
			ax1 = fig.add_subplot(str(23*10+fold_ind+1))
			ax1.plot(test_y, label="original")
			ax1.plot(predicted, label="predicted")
			ax1.legend()
			plt.title('CV Fold-{}'.format(fold_ind+1))
		print('TOTAL MSE: {}'.format(total_err))
		plt.show()

