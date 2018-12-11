import unittest
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


class TestLabExercise(unittest.TestCase):
    def setUp(self):
        self.zooData = pd.read_csv('colonTumor.csv')

    def test_for_None(self):
        self.assertFalse(self.zooData is None)


def nearestNeighbour(X, Y, criteria):
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=1), X, Y, cv=10)
    print ''
    print('Nearest neighbour 10CV Accuracy of ' + criteria + ' : %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    print ''


def PrincipleComponentMethod(X, Y):
    plotX = []
    plotY = []
    for i in range(1, 10):
        pca = PCA(n_components=i)
        pca.fit(X)
        tempX = pca.transform(X)
        nearestNeighbour(tempX, Y, str(i) + ' th Principle Component')
        if i == 2:
            plotX = tempX[:, 0]
            plotY = tempX[:, 1]

    pca = PCA(n_components=50)
    pca.fit(X)
    tempX = pca.transform(X)
    nearestNeighbour(tempX, Y, '50 th Principle Component')

    pca = PCA(n_components=100)
    pca.fit(X)
    tempX = pca.transform(X)
    nearestNeighbour(tempX, Y, '100 th Principle Component')

    plt.scatter(plotX, plotY, edgecolors='k')
    plt.show()


def ChiSquareMethod(X, Y):
    plotX = []
    plotY = []
    for i in range(1, 10):
        tempX = SelectKBest(chi2, k=i).fit_transform(X, Y)
        nearestNeighbour(tempX, Y, str(i) + ' th Component (Chi Square)')
        if i == 2:
            plotX = tempX[:, 0]
            plotY = tempX[:, 1]

    tempX = SelectKBest(chi2, k=50).fit_transform(X, Y)
    nearestNeighbour(tempX, Y, '50 Component (Chi Square)')

    tempX = SelectKBest(chi2, k=100).fit_transform(X, Y)
    nearestNeighbour(tempX, Y, '100 th Component (Chi Square)')

    plt.scatter(plotX, plotY, edgecolors='k')
    plt.show()


if __name__ == "__main__":
    colonTumor = pd.read_csv('colonTumor.csv')

    colonTumor = colonTumor.fillna(colonTumor.mean())
    X = colonTumor.drop(['Class'], axis=1)
    Y = colonTumor['Class']

    nearestNeighbour(X, Y, 'All Data Set')
    '''
    -2-
    Classification Accuracy of All data set using Nearest neighbour 10CV Accuracy of All Data Set : 0.79 (+/- 0.42)
    '''

    PrincipleComponentMethod(X, Y)
    '''
    -3-
    Accuracies as follows :
    Nearest neighbour 10CV Accuracy of 1 th Principle Component : 0.57 (+/- 0.25)
    Nearest neighbour 10CV Accuracy of 2 th Principle Component : 0.61 (+/- 0.24)
    Nearest neighbour 10CV Accuracy of 3 th Principle Component : 0.77 (+/- 0.43)
    Nearest neighbour 10CV Accuracy of 4 th Principle Component : 0.77 (+/- 0.40)
    Nearest neighbour 10CV Accuracy of 5 th Principle Component : 0.78 (+/- 0.42)
    Nearest neighbour 10CV Accuracy of 6 th Principle Component : 0.78 (+/- 0.42)
    Nearest neighbour 10CV Accuracy of 7 th Principle Component : 0.77 (+/- 0.43)
    Nearest neighbour 10CV Accuracy of 8 th Principle Component : 0.77 (+/- 0.43)
    Nearest neighbour 10CV Accuracy of 9 th Principle Component : 0.75 (+/- 0.43)
    Nearest neighbour 10CV Accuracy of 50 th Principle Component : 0.79 (+/- 0.42)
    Nearest neighbour 10CV Accuracy of 100 th Principle Component : 0.79 (+/- 0.42)
    
    Therefore when component is increased accuracy is increase.
    '''

    '''
    -4-
    The 1st component showed a significantly high correlation to the data set when compared to the second. This can
    be observed by the variation of the data set along the first component. (Can be observed by the graph)
    '''

    ChiSquareMethod(X, Y)
    '''
    -5-
    Accuracies as follows :
    Nearest neighbour 10CV Accuracy of 1 th Component (Chi Square) : 0.72 (+/- 0.37)
    Nearest neighbour 10CV Accuracy of 2 th Component (Chi Square) : 0.71 (+/- 0.25)
    Nearest neighbour 10CV Accuracy of 3 th Component (Chi Square) : 0.81 (+/- 0.32)
    Nearest neighbour 10CV Accuracy of 4 th Component (Chi Square) : 0.77 (+/- 0.34)
    Nearest neighbour 10CV Accuracy of 5 th Component (Chi Square) : 0.79 (+/- 0.34)
    Nearest neighbour 10CV Accuracy of 6 th Component (Chi Square) : 0.87 (+/- 0.42)
    Nearest neighbour 10CV Accuracy of 7 th Component (Chi Square) : 0.88 (+/- 0.33)
    Nearest neighbour 10CV Accuracy of 8 th Component (Chi Square) : 0.87 (+/- 0.42)
    Nearest neighbour 10CV Accuracy of 9 th Component (Chi Square) : 0.87 (+/- 0.42)
    Nearest neighbour 10CV Accuracy of 50 Component (Chi Square) : 0.79 (+/- 0.42)
    Nearest neighbour 10CV Accuracy of 100 th Component (Chi Square) : 0.77 (+/- 0.40)
    
    Therefore,the number of selected components are increasing, using the chi squred measure, the accuracy increased
    '''

    '''
    -6-
    All the accuracies of PCA are less than the accuracy for 1-nearest neighbor with 10-fold cross validation on the
    unfiltered data set. But 7 components obtained from chi square measure showed the highest accuracy for the given
    data set. In Generally PAC accuracies lies in range of 70% to 80% and Chi Square accuracies lies in range of 
    80% to 90%. Therefore, extracting 7 components using the chi squared measure is the best suited method for the given
     data set. 
    '''

    unittest.main()
