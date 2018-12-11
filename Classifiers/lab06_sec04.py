'''
E/13/058
Try out section co544 lab 06
Classification
Date : 24/08/2017
'''

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class testAccuracy:
    def training_and_testing(self, X, Y, clf):
        # 4.5 Spliting dataset as training and testing

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.333, random_state=0)
        clf.fit(X_train, Y_train)
        print('Training Accuracy using Spliting dataset as training and testing: ', clf.score(X_test, Y_test))

    def cross_val_score(self, X, Y, clf):
        # 4.6 Use of Cross -val -score helper function fro cross validation
        print ('Using Cross -val -score helper function for cross validation ')
        scores = cross_val_score(clf, X, Y, cv=10)
        print(scores)
        print("10CV Accuracy : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def confusion_matrix(self, X, Y, clf):
        # 4.7 Confusion matrix to get the details of the classification

        print ('Using Confusion matrix to get the details of the classification ')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.333, random_state=0)
        Y_pred = clf.fit(X_train, Y_train).predict(X_test)
        print(confusion_matrix(Y_test, Y_pred))

    def precission_recall(self, X, Y, clf):
        print ('precision recall details')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.333, random_state=0)
        Y_pred = clf.fit(X_train, Y_train).predict(X_test)
        target_names = ['setosa', 'versicolor', 'virginica']
        print(classification_report(Y_test, Y_pred, target_names=target_names))


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    accuracy = testAccuracy()

    print('__________Gausian Naive Bayes__________')

    clf = GaussianNB()
    clf.fit(X, Y)
    print('Training Accuracy using Gausian Naive Bayes: ', clf.score(X, Y))
    accuracy.training_and_testing(X, Y, clf)
    accuracy.cross_val_score(X, Y, clf)
    accuracy.confusion_matrix(X, Y, clf)
    accuracy.precission_recall(X, Y, clf)

    print('_________Multinomial Naive Bayes_________')

    clf = MultinomialNB()
    clf.fit(X, Y)
    print('Training Accuracy using Multinomial Naive Bayes: ', clf.score(X, Y))
    accuracy.training_and_testing(X, Y, clf)
    accuracy.cross_val_score(X, Y, clf)
    accuracy.confusion_matrix(X, Y, clf)
    accuracy.precission_recall(X, Y, clf)

    print('__________Nearest Neighbor__________')

    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, Y)
    print('Training Accuracy using Nearest Neighbor: ', clf.score(X, Y))

    accuracy.training_and_testing(X, Y, clf)
    accuracy.cross_val_score(X, Y, clf)
    accuracy.confusion_matrix(X, Y, clf)
    accuracy.precission_recall(X, Y, clf)

    print('________Support Vector Machine_________')

    clf = svm.SVC(kernel='linear', C=1, gamma=1).fit(X, Y)
    clf.fit(X, Y)
    print('Training Accuracy using Support Vector Machine: ', clf.score(X, Y))
    accuracy.training_and_testing(X, Y, clf)
    accuracy.cross_val_score(X, Y, clf)
    accuracy.confusion_matrix(X, Y, clf)
    accuracy.precission_recall(X, Y, clf)
