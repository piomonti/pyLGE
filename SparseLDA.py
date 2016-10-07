### Sparse LDA based on optimal scoring
#
# See Clemmensen (2011) for reference.
# Basic idea is recast LDA as a regression 
# problem which can then be easily solved
#
#

import numpy
from sklearn.linear_model import ElasticNet

def SparseLDA(Y, X, alpha, l1_ratio, tol=0.1, maxIter=50):
    """
    Implementation of Sprase LDA as in Clemmensen (2011)
    
    INPUT:
	 - Y: matrix of size (n,K) where n is number of obs and K is number of classes. Each row of Y is an indicator variable
	 - X: matrix of resposne variables, size (n,p)
	 - alpha, l1_ratio: amount of regularisation and ratio of l1 to l2 (as in sklearn documentation)
	 - tol: convergence tolerance
	 - maxIter: maximum number of iterations on each discriminant vector
    
    OUTPUT:
	 - B: matrix of K-1 discrimination vectors, size (n, K-1)

    """
    
    K = Y.shape[1]
    
    D = (1./Y.shape[0]) * numpy.dot( Y.transpose(), Y)
    Dinv = numpy.linalg.inv(D)
    Q = numpy.ones((K,1))
    
    EnetModel = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept = False, normalize=False)
    
    for k in range(K-1):
	#print "Estimating the " + str(k+1) + "th discriminant vector"
	
	# randomly initialise theta_k
	theta_k = numpy.random.randn(K).reshape((K,1))
	# project onto orthogonal space of Q
	theta_k = numpy.dot( numpy.eye(K) - numpy.dot(Q, numpy.dot(Q.transpose(), D)), theta_k)
	# normalise:
	theta_k /= numpy.sqrt( numpy.dot(theta_k.transpose(), numpy.dot(D, theta_k) ))
	
	conv = False
	iter_ = 0
	beta_k_old = 0
	
	while ((conv==False) & (iter_<maxIter)):
	    beta_k = EnetModel.fit(X=X, y=numpy.dot(Y, theta_k)).coef_
	    theta_k = (numpy.eye(K) - numpy.dot(Q, numpy.dot(Q.transpose(), D)) ).dot( numpy.dot(Dinv, numpy.dot(Y.transpose(), numpy.dot(X, beta_k)))) 
	    
	    # normalise:
	    theta_k /= numpy.sqrt( numpy.dot(theta_k.transpose(), numpy.dot(D, theta_k) ))
	    
	    # check convergence:
	    if abs(beta_k-beta_k_old).sum() < tol:
		conv = True
	    else:
		iter_ += 1
		beta_k_old = numpy.copy(beta_k)
		
	# append results:
	#print "here"
	Q = numpy.hstack((Q, theta_k.reshape((K,1))))
	#print "here"
	if k==0:
	    B = numpy.copy(beta_k).reshape((X.shape[1],1))
	else:
	    B = numpy.hstack((B, beta_k.reshape((X.shape[1],1)) ))
	#print "here"
    
    return B
	    
    
    
