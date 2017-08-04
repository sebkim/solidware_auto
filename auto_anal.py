import pandas as pd
from sklearn.preprocessing import Imputer
import numpy as np

col_names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', \
'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', \
'length', 'width', 'height', 'curb-weight', 'engine-type', \
'num-of-cylinders','engine-size', 'fuel-system', 'bore', 'stroke', \
'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', \
'price']

col_types = ['int', 'int', 'str', 'str', 'str', \
'int', 'str', 'str', 'str', 'float', \
'float', 'float', 'float', 'int', 'str', \
'int', 'int', 'str', 'float', 'float', \
'float', 'int', 'int', 'int', 'int', \
'int']

automobile = pd.read_csv('data/imports-85.data', names=col_names)
automobile = automobile.replace('?', np.nan)


imp = Imputer(missing_values='NaN', strategy="mean", axis=0)

print automobile.dtypes
print automobile.isnull().sum()
print automobile
# print automobile[]
# xx = imp.fit_transform(automobile)


# X = automobile.drop(['normalized-losses'], axis=1)
# Y = automobile['normalized-losses']


