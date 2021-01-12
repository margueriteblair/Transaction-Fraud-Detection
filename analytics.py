
#import libraries for project:
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split


def train_model():

    dataset = pd.read_csv('~/Desktop/PS_20174392719_1491204439457_log.csv')

    print(dataset.shape)

    X = dataset.iloc[:, [1, 2, 4, 5, 7, 8]].values
    y = dataset.iloc[:, 9].values
    y_target = dataset['isFraud']

    # #below, we're going to transform the data into a binary value from the csv
    # from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # label_encoder = LabelEncoder()
    # X[:, 0] = label_encoder.fit_transform(X[:, 0])

    # #then we retransform into numbers b/c OneHotEncoder can't use strings
    # one_hot_encoder = OneHotEncoder(categorial_features=[0])
    # X = one_hot_encoder.fit_transform(X).toarray()

    #create the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #feature scaling: all values are within similar range
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    #now fit multiple linear regressions to the training set
    from sklearn.linear_model import LogisticRegression
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_test, y_pred);
    return conf_mat


