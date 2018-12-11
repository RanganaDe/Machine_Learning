import pandas as pd
from sklearn import neighbors, svm, tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


def trainingSet(X, Y, clf):
    clf.fit(X, Y)
    print'Training Accuracy    : ', clf.score(X, Y)


def trainingAndTesting(X, Y, clf):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.333, train_size=0.667, random_state=0)
    clf.fit(X_train, Y_train)
    print 'Test Accuracy      : ', clf.score(X_test, Y_test)


def crossValidation(X, Y, clf):
    scores = cross_val_score(clf, X, Y, cv=10)
    print('10CV Accuracy    : %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))


def confusionMatrix(X, Y, clf):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.333, train_size=0.667, random_state=0)
    Y_predict = clf.fit(X_train, Y_train).predict(X_test)
    print 'Confusion matrix :'
    print (confusion_matrix(Y_test, Y_predict))


def exerciseTwo(X, Y):
    print ''
    print '------------------------------- Exercise two -------------------------------'
    print '--------------------------- Decision Tree Classifier ---------------------------'
    clf = tree.DecisionTreeClassifier()
    trainingSet(X, Y, clf)
    trainingAndTesting(X, Y, clf)
    crossValidation(X, Y, clf)


def exerciseThree(X, Y):
    print ''
    print '------------------------------- Exercise Three -------------------------------'
    print '------------------- Confusion Matrix of classifiers done in Lab -------------------'
    print ''
    print '------------------------------- Gaussian Naive Bayes -------------------------------'
    confusionMatrix(X, Y, GaussianNB())
    print ''
    print '------------------------------- Multinomial Naive Bayes -------------------------------'
    confusionMatrix(X, Y, MultinomialNB())
    print ''
    print '------------------------------- K-Nearest Neighbor -------------------------------'
    confusionMatrix(X, Y, neighbors.KNeighborsClassifier(n_neighbors=1))
    print ''
    print '------------------------------- Support Vector Machine -------------------------------'
    confusionMatrix(X, Y, svm.SVC(kernel='linear', C=1, gamma=1))


def exerciseFour(X, Y):
    print ''
    print '------------------------------- Exercise Four -------------------------------'
    print '------------------- 10-fold cross validation accuracies of the classifiers done in Lab -------------------'
    print ''
    print '------------------------------- Gaussian Naive Bayes -------------------------------'
    crossValidation(X, Y, GaussianNB())
    print ''
    print '------------------------------- Multinomial Naive Bayes -------------------------------'
    crossValidation(X, Y, MultinomialNB())
    print ''
    print '------------------------------- K-Nearest Neighbor -------------------------------'
    crossValidation(X, Y, neighbors.KNeighborsClassifier(n_neighbors=1))
    print ''
    print '------------------------------- Support Vector Machine -------------------------------'
    crossValidation(X, Y, svm.SVC(kernel='linear', C=1, gamma=1))


def exerciseFive(X, Y):
    print ''
    print '------------------------------- Exercise Five -------------------------------'
    print '------------------------------- Bernoulli Naive Bayes -------------------------------'
    confusionMatrix(X, Y, BernoulliNB())
    crossValidation(X, Y, BernoulliNB())
    trainingAndTesting(X, Y, BernoulliNB())
    trainingSet(X, Y, BernoulliNB())


if __name__ == "__main__":
    # Read CSV and load zoo data set
    zooData = pd.read_csv('zooData.csv')

    # Exercise 1
    print zooData.head()

    if sum(zooData.isnull().values.ravel()) != 0:
        zooData = zooData.fillna(zooData.mean())

    X = map(list, zooData[zooData.columns.difference(['animalName', 'type'])].values)
    Y = zooData.type.T.tolist()

    exerciseTwo(X, Y)

    exerciseThree(X, Y)

    exerciseFour(X, Y)

    exerciseFive(X, Y)


