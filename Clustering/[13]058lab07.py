import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

'''
E/13/058
LAB 07
Clustering
'''

if __name__ == '__main__':

    # Read the data from .csv file

    X = pd.read_csv('breaset-cancer.csv')
    X = X.fillna(X.mean())

    # delete the country data
    del X['COUNTRY']

    z = X.iloc[:, 0:3]

    # estimate clusters using Kmean

    est = KMeans(n_clusters=3, random_state=0).fit(z)
    labels = est.labels_

    # plot data set using 6 sub plots
    x = X.iloc[:, 0]
    y = X.iloc[:, 1]
    axis = plt.subplot(231)
    plt.scatter(x, y, edgecolors='k', c=labels)

    x = X.iloc[:, 0]
    y = X.iloc[:, 2]
    axis = plt.subplot(232)
    plt.scatter(x, y, edgecolors='k', c=labels)

    x = X.iloc[:, 0]
    y = X.iloc[:, 3]
    axis = plt.subplot(233)
    plt.scatter(x, y, edgecolors='k', c=labels)

    x = X.iloc[:, 1]
    y = X.iloc[:, 2]
    axis = plt.subplot(234)
    plt.scatter(x, y, edgecolors='k', c=labels)

    x = X.iloc[:, 1]
    y = X.iloc[:, 3]
    axis = plt.subplot(235)
    plt.scatter(x, y, edgecolors='k', c=labels)

    x = X.iloc[:, 2]
    y = X.iloc[:, 3]
    axis = plt.subplot(236)
    plt.scatter(x, y, edgecolors='k', c=labels)

    plt.show()

    '''
    Heirarchical Clustering
    '''
    # generate linkage matrix
    Z = linkage(X, 'average')
    plt.figure(figsize=(25, 10))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8, )
    dendrogram(Z, truncate_mode='lastp', leaf_rotation=90, leaf_font_size=8)

    plt.axhline(y=10, c='k')
    plt.show()

'''
2.
Random State in Kmean's function

random_state is either int, RandomState instance or None. It is optional and at default set to None
If it is set to int ,it means that it is the seed used by the random number generator and if it is a random state instance then
random_state is the random number generator and if its None then the random number generator is the Randomstate instance used by np.random


3.
We can see the separation of the clusters in first three sub plots more clearly than the latter three sub plots.

5.
K means algorithm is more appopraite in oaccasions where the the number of clusters is known as it clusters only into the provided number of clusters.
Heirachical clustering more appropiate in occasions where there is no defined number of clusters and number of clusters is hard to decide.
dendogram in heirachical clustering allows to find the suitable number of clustering in those occasions.


'''