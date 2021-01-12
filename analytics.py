
#import libraries for project:
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def train_model():

    dataset = pd.read_csv('~/Desktop/PS_20174392719_1491204439457_log.csv')

    print(dataset.shape)

    X = dataset.iloc[:, [1, 2, 4, 5, 7, 8]].values
    y = dataset.iloc[:, 9].values
    y_target = dataset['isFraud']

    #create the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #now fit multiple linear regressions to the training set
    regression = LogisticRegression()
    regression.fit(X_train, y_train)
    y_pred = regression.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred);
    return conf_mat


