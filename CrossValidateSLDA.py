## Run and cross validate Sparse LDA
#
# We run 10-fold cross validation
#

import numpy, os, pandas, math
os.chdir('/media/ricardo/1401-1FFE/Documents/Writing/LGE/v1/Code')
from SparseLDA import *
from OneVsRestLDAClassifiers import *
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression 
from random import shuffle

def CVsparseLDA(pred, resp, l1range, Kfold=10):
    """
    INPUT:
	- pred: predictor matrix
	- resp: response matrix (each row an indicator variable) or a indicator vector...
	- l1range: list of values to cross validate over
	- Kfold: number of folds to cross validate
    
    """
    
    # divide into K random folds: 
    ii = map(lambda x: int(math.floor(x)), list(numpy.linspace(0, resp.shape[0]-1, Kfold+1)) )
    #index = ( list(numpy.linspace(0, resp.shape[0]-1, Kfold+1) ))
    index = list(range(resp.shape[0]-1))
    shuffle(index)

    # matrix to store results:
    AUCscores = numpy.zeros(( len(l1range) , Kfold ))
    
    # confusion matrix (one confusion for each l1 value):
    if len(resp.shape)==1:
	# only a vector has been provided, will measure performance based on confusion matrix
        confusion = numpy.zeros((len(l1range), Kfold))
        p = len(set(resp))
	ConfOnly = True
    else:
        confusion = numpy.zeros((len(l1range), 2,2))
	ConfOnly = False
    
    # start cross validating:
    for k in range(Kfold):
	if k == 0:
	    TrainID = index[ ii[1]:]
	    TestID = index[ii[0]:ii[1]]
	    #TrainID = map( int, numpy.linspace(ii[k+1], ii[-1], ii[-1]-ii[k+1]+1))
	    #TestID = map( int, numpy.linspace(ii[k], ii[k+1], ii[k+1]-ii[k]))
	    #print "Test size: " + str(len(TestID)) + ". Train size: " + str(len(TrainID)) # for debugging
	if k == Kfold-1:
	    TrainID = index[ :ii[k] ]
	    TestID = index[ ii[k]: ]
	    #TestID = map( int, numpy.linspace(ii[k], ii[-1], ii[-1]-ii[k]+1))
	    #TrainID = map( int, numpy.linspace(ii[0], ii[k], ii[k]-ii[0]+1))
	    #print "Test size: " + str(len(TestID)) + ". Train size: " + str(len(TrainID)) # for debugging
	else:
	    TrainID = index[ :ii[k] ] + index[ ii[k+1]: ]
	    TestID = index[ ii[k]:ii[k+1]  ]
	    #TrainID = map( int,  numpy.concatenate(( numpy.linspace(ii[0],ii[k],ii[k]-ii[0]+1), numpy.linspace(ii[k+1],ii[-1],ii[-1]-ii[k+1]+1)) ) )
	    #TestID = map( int, numpy.linspace(ii[k], ii[k+1], ii[k+1]-ii[k]+1))
	    #print "Test size: " + str(len(TestID)) + ". Train size: " + str(len(TrainID)) # for debugging
	
	# build test/train datasets:
	TrainData = pred[TrainID,:]
	TestData = pred[TestID,:]
	# and responses:
	if ConfOnly:
            TrainResp = numpy.zeros(( len(TrainID), p  )) #resp[TrainID]
            TestResp = numpy.zeros(( len(TestID), p  )) # resp[TestID]
	    counterBuilder = 0
	    for j in set(resp):
		TrainResp[:, counterBuilder][ resp[TrainID]==j] = 1
		TestResp[:, counterBuilder][ resp[TestID]==j] = 1
		counterBuilder += 1
	else:
            TrainResp = resp[TrainID,:]
            TestResp = resp[TestID,:]

	# fit on training data:
	for l in range(len(l1range)):
	    SLDA = SparseLDA(Y=TrainResp, X=TrainData, alpha=l1range[l], l1_ratio=1)
	    Score = numpy.dot(TestData, SLDA)

	    if ConfOnly:
		# just consider confusion scores
		C, lp = OneVsAll(resp[TestID], Score)
		confusion[l, k] = numpy.diagonal(C).sum()/float(len(TestID))
	    else:
     	        fpr1, tpr1, thresholds = roc_curve(TestResp[:,0], Score)
	        fpr2, tpr2, thresholds = roc_curve(TestResp[:,0], Score*-1) # sometimes need to flip sign - altho you would think it would happen automatically...
	        AUCscores[l,k] = max(auc(fpr1, tpr1), auc(fpr2, tpr2))
	    
	        # also get TPs and FPs
	        logreg = LogisticRegression(C=1e5, fit_intercept=False) # use log reg to choose best cut-off threshold (huge C implies no regularisation)
	        logreg.fit(Score, TestResp[:,0])
	        predicted = logreg.predict(Score)
	        confusion[l,:,:][0,0] += ((predicted==1) & (TestResp[:,0]==1)).sum()		#((Score.reshape(-1) > 0) & (TestResp[:,0] > 0)*1).sum() # true positives
	        confusion[l,:,:][1,0] += ((predicted==1) & (TestResp[:,0]==0)).sum()		#((Score.reshape(-1) > 0) & (TestResp[:,0] <= 0)*1).sum() # false positives
	        confusion[l,:,:][0,1] += ((predicted==0) & (TestResp[:,0]==1)).sum()		#((Score.reshape(-1) <= 0) & (TestResp[:,0] > 0)*1).sum() # false negative
	        confusion[l,:,:][1,1] += ((predicted==0) & (TestResp[:,0]==0)).sum()		#((Score.reshape(-1) <= 0) & (TestResp[:,0] <= 0)*1).sum() # true negative
	    
    AUCmin1sd = numpy.copy(AUCscores)
    AUCmin1sd = AUCmin1sd - AUCmin1sd.std(axis=1).reshape(( len(l1range),1 ))
    
    # now also return with 1 std dev removed, to penalised inconsistent/high varying choices of l1
    if ConfOnly:
        return confusion, l1range
    else:
	return AUCscores, AUCmin1sd, l1range, confusion
