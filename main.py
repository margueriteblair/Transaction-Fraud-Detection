
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
one_hot_encoder = OneHotEncoder(categorial_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()

#now wesplit the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#feature scaling: all values are within similar range


