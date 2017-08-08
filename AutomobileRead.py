import pandas as pd
import numpy as np
from sklearn import preprocessing
from helper import *
from fancyimpute import NuclearNormMinimization

class AutomobileRead(object):
	def __init__(self, default_read='data/imports-85.data'):
		self.col_names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', \
		'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', \
		'length', 'width', 'height', 'curb-weight', 'engine-type', \
		'num-of-cylinders','engine-size', 'fuel-system', 'bore', 'stroke', \
		'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', \
		'price']
		self.col_types = ['int', 'int', 'str', 'str', 'str', \
		'int', 'str', 'str', 'str', 'float', \
		'float', 'float', 'float', 'int', 'str', \
		'int', 'int', 'str', 'float', 'float', \
		'float', 'int', 'int', 'int', 'int', \
		'int']
		# read csv
		self.automobile = pd.read_csv('data/imports-85.data', names=self.col_names)
		# ignore symboling column
		self.automobile = self.automobile.drop('symboling', axis=1); self.col_names.pop(0); self.col_types.pop(0)
		# apply text2int for num-of-doors and num-of-cylinders
		for each in ['num-of-doors', 'num-of-cylinders']:
			self.automobile[each] = self.automobile[each].apply(text2int)
	# Must be categorized before using GetImputedDataframe
	def GetImputedDataframe(self, df, impute_type='mean'):
		df = df.copy()
		# impute missing values
		if impute_type=='mean':
			null_sum = df.replace('?', np.nan).isnull().sum()
			null_col = [k for k,v in null_sum.iteritems() if v!=0]
			for each_col_ind, each_col in enumerate(self.col_names):
				if each_col in null_col and self.col_types[each_col_ind]!='str':
					df[each_col] = mean_impute(df, each_col, data_type=self.col_types[each_col_ind])
		elif impute_type=='nnm':
			df = df.replace('?', np.nan)
			df = pd.DataFrame(NuclearNormMinimization().complete(df), columns = self.col_names)
		else:
			raise Exception()
		return df
	def DoCategorizeFrom(self, df):
		df = df.copy()
		# label encoding from categorical values to integer
		for each_col_ind, each_col in enumerate(self.col_names):
			if self.col_types[each_col_ind] == 'str':
				le = preprocessing.LabelEncoder()
				le.fit(df[each_col])
				df[each_col] = le.transform(df[each_col])
		return df

