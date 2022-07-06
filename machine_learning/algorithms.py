from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import printcode


def split_data(data, test_size, output, independent):
    test_size = int(test_size)/100
    y = data[output]
    X = data.drop(output, axis=1)
    if independent != '':
        independent = ' '.join(independent.split())
        independent = list(independent.split(", "))
        X = data[independent]
        print(X)
        print(independent)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    printcode.printcode_splitdata(independent, output, test_size)
    return(X_train, X_test, y_train, y_test)


def linear_regression(split):
    X_train, X_test, y_train, y_test = split
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    printcode.printcode_linearregression()
    return y_test, y_pred


def logistic_regression(split):
    X_train, X_test, y_train, y_test = split
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    printcode.printcode_logisticregression()
    return y_test, y_pred


def naive_bayes(split):
    X_train, X_test, y_train, y_test = split
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    printcode.printcode_naivebayes()
    return y_test, y_pred


def svc(split):
    X_train, X_test, y_train, y_test = split
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    printcode.printcode_svc()
    return y_test, y_pred
