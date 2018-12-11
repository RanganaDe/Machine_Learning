'''
E/13/058
Decision Trees
Date : 8/15/2017
'''

import sklearn
from sklearn import tree
import os

import pydotplus
import pandas as pd

import unittest


class breastCancerPrediction:

    def __init__(self):
        # read dataset
        testData = pd.read_csv('breaset-cancer.csv')
        # remove country data
        del testData['COUNTRY']
        #Handle missing values
        testData = testData.fillna(testData.mean())
        self.testData = testData
        responseVariable = testData['BREASTCANCERPER100TH']
        self.responseVariable = responseVariable

        del testData['BREASTCANCERPER100TH']

        # devide the values by 20 and convert rest to nominal values
        for i in range(responseVariable.size):
            if self.responseVariable.values[i] <= 20:
                self.responseVariable.values[i] = 0
            else:
                self.responseVariable.values[i] = 1

        for i in range(self.testData.shape[1]):
            k = self.testData.values[:, i].mean()
        for j in range(self.testData.values[:, i].size):
            if self.testData.values[:, i][j] <= k:
                self.testData.values[:, i][j] = 0
            else:
                self.testData.values[:, i][j] = 1

        # split data

        split_val = self.testData.shape[0] / 3

        self.train_data = self.testData.iloc[split_val:, :]
        self.test_data = self.testData.iloc[:split_val, :]
        self.train_responseVariable = self.responseVariable.iloc[split_val:]
        self.test_responseVariable = self.responseVariable.iloc[:split_val:]

    def predict(self):

        #classify
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.train_data, self.train_responseVariable)

        #generate pdf
        with open("breast_cancer.dot", "w") as f:
            f = tree.export_graphviz(clf, out_file=f)
        os.unlink('breast_cancer.dot')

        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("breast_cancer.pdf")

        # Predict values

        return sklearn.metrics.accuracy_score(self.test_responseVariable, clf.predict(self.test_data))


if __name__ == '__main__':

    cancerPredict = breastCancerPrediction()
    print 'Prediction Accuracy for test data set : ', cancerPredict.predict()
