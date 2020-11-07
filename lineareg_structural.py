# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 10:24:41 2018

@author: Lenovo
"""

import statsmodels.api as sm # import statsmodels 
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from matplotlib import style
from pylab import * 
dataset=pd.read_csv("linkin.csv").values

#split dependent variable and independent variable

Y=dataset[:,4]

X=dataset[:,0:5] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(Y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics.

model.summary()
lm = linear_model.LinearRegression()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.5,random_state=4)
model = lm.fit(x_train,y_train)
lm.intercept_
print('coeff:\n',lm.coef_)
print('mean error:\n',np.mean((lm.predict(x_test)-y_test)**2))
print('variance:\n',lm.score(x_test,y_test))
predictions = lm.predict(x_test)
pred=sorted(predictions,reverse=True)

myList = list(np.around(np.array(pred),5))
print(myList)
style.use('bmh')
plt.scatter(lm.predict(x_test),y_test)
plt.show()



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

intercept = -2.0
beta = 5.0
n_samples = 1000
regularization = 1e30
X = np.random.normal(size=(n_samples,1))
linepred = intercept + (X*beta)
prob = np.exp(linepred) / (1.0 + np.exp(linepred))
y = (np.random.uniform(size=X.shape) < prob).astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print ("Percent of class 1: ", sum(y)/len(y))
plt.figure()
plt.plot(X,prob, '.', color='blue', label='model', markersize=0.5)

clf = LogisticRegression(solver='liblinear', tol=1e-10, max_iter=10000, C=regularization);
clf.fit(X_train, y_train.ravel());
print ("Coeff: {}, Intercept: {}".format(clf.coef_, clf.intercept_))
print ("Score over training: ", clf.score(X_train, y_train))
print ("Score over testing: ", clf.score(X_test, y_test))
plt.plot(X, clf.predict_proba(X)[:,1], '.', color='red', label='clf', markersize=0.5)
tot_score = clf.score(X, y)
plt.title("Score: {}".format(tot_score))
plt.legend(loc='best');
plt.show()
