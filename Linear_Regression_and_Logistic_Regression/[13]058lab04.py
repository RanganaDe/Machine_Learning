'''
E/13/058
Linear Regression and Logistic Regression
'''

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

iris=datasets.load_iris()
X=iris["data"][:,3:]
#print X
#if target value ==2 then return 1,else 0
Y=(iris["target"]==2).astype(np.int)
#print Y


#train a logistic regression model

X_train,X_val,Y_train,y_val=train_test_split(X,Y,test_size=0.2)
log_reg=LogisticRegression()
log_reg.fit(X_train,Y_train)
Y_proba=log_reg.predict(X_val)
print 'logistic regression result:\n', Y_proba

#train a linear regression model


lin_reg=LinearRegression()
lin_reg.fit(X_train,Y_train)
y_proba=lin_reg.predict(X_val)
print'Linear Regression result:\n', y_proba
