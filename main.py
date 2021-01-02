
#import libraries for project:
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import sklearn

dataset = pd.read_csv('~/Desktop/PS_20174392719_1491204439457_log.csv')

print(dataset.shape)

X = dataset.iloc[:, [1, 2, 4, 5, 7, 8]].values
Y = dataset.iloc[:, 9].values

#below, we're going to transform the data into a binary value from the csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X[:, 0] = label_encoder.fit_transform(X[:, 0])

#then we retransform into numbers b/c OneHotEncoder can't use strings


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
