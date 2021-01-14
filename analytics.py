
#import libraries for project:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#we want to encompass all  of this within a method so it can easily be called from our router
def train_model():

    #importing the dataset
    dataset = pd.read_csv('~/Desktop/PS_20174392719_1491204439457_log.csv')

    #printing out the number of rows and columns of our dataset
    print(dataset.shape)

    #X is our input data and we'll be using rows 2, 4, 5, 7, 8 to detect fraud in this instance
    #It's common practice to name the input dataset 'X'
    X = dataset.iloc[:, [2, 4, 5, 7, 8]].values

    #It's common praticeto name the output dataset 'y' lowercase.
    #y is our output target dataset b/c it's got the verdict whether or not a transaction is or isn't actually fraud
    y = dataset.iloc[:, 9].values
    y_target = dataset['isFraud']

    #create the training set and the test set
    #we know train_test_split() returns a tuple, so we can get all of these variables at once from calling it on our data
    #train_test_split(inputSet, outputSet, test_size, random_state, stratify)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0, stratify=y_target)

    #now fit multiple linear regressions to the training set
    regression = LogisticRegression(random_state=0)
    regression.fit(X_train, y_train)
    y_pred = regression.predict(X_test)


    #using our test set results, we can use a confusion matrix to compare the results our model came up with and our actual y_test
    conf_mat = confusion_matrix(y_test, y_pred);
    # print(conf_mat)

    true_non_fraud, false_non_fraud, true_fraud, false_fraud = confusion_matrix(y_test, y_pred).ravel()

    accuracy = round(metrics.accuracy_score(y_test, y_pred), 4)*100
    precision = round(metrics.precision_score(y_test, y_pred), 4)*100
    recall = round(metrics.recall_score(y_test, y_pred), 4)*100

    print("Accuracy:     ", accuracy)
    print("Precision:    ", precision)
    print("Recall:       ", recall)

    result = [
        {
            'Test data Size': y_test.size,
            'Non-Fradulent predicted True': true_non_fraud.item(),
            'Non-Fradulent predicted false': false_non_fraud.item(),
            'Fradulent predicted True': true_fraud.item(),
            'Fradulent predicted false': false_fraud.item()
        },

        {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        }
    ]

    return result


