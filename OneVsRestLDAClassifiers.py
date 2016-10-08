### Build 1 vs rest LDA classifiers
#
#
#
#

import numpy, os
from SparseLDA import *
from sklearn.linear_model import LogisticRegression # will perform one vs all for us
from sklearn.metrics import confusion_matrix

def OneVsAll(Class, Predictor):
    """Fit a one vs all classifier using Logistic regression
	
    INPUT:
	- Class: classification matrix
	- Predictor: predictor matrix

    """
    model = LogisticRegression(C=100, fit_intercept=True)
    model.fit(Predictor, Class)

    # get confusion matrix:
    C = confusion_matrix( Class, model.predict(Predictor))

    # get log-likelihood also:
    logProb = model.predict_log_proba(Predictor)
    logProbSum = 0
    for i in range(logProb.shape[0]):
	logProbSum += logProb[i,:][int(Class[i]-1)]

    return C, logProbSum

