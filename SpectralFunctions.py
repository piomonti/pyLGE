## Spectral Functions!
#
# Collection of function to build Laplacian matricies etc
#
#

import numpy

def normalizePrecision(P):
    """Normalise precision matrix!"""
    
    P2 = numpy.copy(P)
    P2 /= numpy.outer(numpy.sqrt(numpy.diagonal(P2)), numpy.sqrt(numpy.diagonal(P2)))
    
    return P2
    

def getL(A, Binary=False):
    """Given an adjacency matrix, calculate the Laplacian
    
    INPUT:
	 - A: square, symmetric adjacency matrix
	 - Binary: boolean to indicate if we should take binary or weighted Laplacian
    
    OUTPUT:
	 - L: square, symmetric Laplacian matrix
    
    *with fix
    """
    
    if Binary:
	BAdj = (A!=0)*1
	numpy.fill_diagonal(BAdj,0) # remove diagaonal entries
	D = numpy.diag( BAdj.sum(axis=0) )
	L = D - BAdj
	
	# we wont normalise for now, because some nodes have degree 0 ==> some infinite values...
	#Dnorm = numpy.diag( 1./BAdj.sum(axis=0) )
	#L = numpy.dot(numpy.dot( numpy.sqrt(D), (D - BAdj)), numpy.sqrt(D))
    else:
	BAdj = abs(A) # remove negative values for now (make sure this makes sense!)
	# normalise to make diagonals 1:
	#BAdj /= numpy.outer( numpy.sqrt(numpy.diagonal(BAdj)), numpy.sqrt(numpy.diagonal(BAdj)) ) 
	#numpy.fill_diagonal(BAdj,0) # remove diagaonal entries
	D = numpy.diag( BAdj.sum(axis=0) )
	#Dhalf = numpy.linalg.inv(numpy.sqrt(D))
	L = D - BAdj
	
	L /= (numpy.outer( numpy.sqrt(numpy.diagonal(L)), numpy.sqrt(numpy.diagonal(L)) ) +1e-5)
	
    
    return L

def getLargestSmallestE(L):
    """Function to get largest & smallest non-zero eigenvalues
    Actually also calculate a host of other things taken from:
    http://www.sciencedirect.com/science/article/pii/S0031320309000065
    
    INPUT:
	 - L: square, symmetric Laplacian matrix
    
    OUTPUT:
	 - m: list with largest & smallest evalue of Laplacian & whole bunch of extra stuff
    """
    
    E = numpy.linalg.eigvals(L)
    E = E[ E.nonzero() ] # remove zeros
    E = E[ E>0] # to remove negatives
    
    return [E.max(), E.min(), numpy.log(numpy.prod(1/E))]

def TopNevectors(L, n=1):
    """Function to return n eigenvectors corresponding to largest eigenvalues
    INPUT:
	 - L: square, symmetric Laplacian matrix of shape (p,p)
    
    OUTPUT:
	 - V: array of shape (p,n) wher each column is an eigenvector
    """
    
    E = numpy.linalg.eig(L)
    E[0][E[0]<0] = 0
    ii = numpy.argsort(-E[0])    
    
    V = E[1][:,ii[:n]]
    for i in range(n):
	# mutliply by corresponding evalue:
	V[:,i] *= E[0][ii][i]
	# adjust to make sure eigenvectors are not flipped:
	if sum(V[:,i]<0) > V.shape[0]/2:
	    V[:,i] *= -1
    
    return V
    
    
def DecomposeLintoS(L, n=0):
    """
    Decompose all evectors into symmetric polynomial values
    
    INPUT:
	 - L: square, symmetric Laplacian matrix
	 - n: number of leading S polys to take - if n=0 we take all of them
	 
    OUTPUT:
	 - vector length L.shape[0]**2 of symmetric poly values for each evector
	 
    
    """
    
    if n==0:
	n = L.shape[0]
    
    r = numpy.zeros((n*L.shape[0])) #resulting vector of S polynomials
    
    E = numpy.linalg.eig(L)
    E[0][E[0]<0] = 0
    ii = numpy.argsort(-E[0])
    
    # e = E[0][ii]
    v = E[1][:,ii]*(E[0][ii]) # ordered according to largest evalue & multiplied by corresponding evalue
    for i in range(v.shape[0]):
	# get S poly values for each eigenvector
	r[(i*n):((i+1)*n) ] =  getSpoly(v[:,i])[0:n]
	
    return r
    

def getSpoly(v):
    """
    Calculate values for SYMMETRIC POLYNOMIALS as described in:
    http://eprints.whiterose.ac.uk/1994/1/hancocker10.pdf

    INPUT:
	 - v: eigenvector
	 
    OUTPUT:
	 - symmetric poly values for v
    
    
    """
    
    
    # some last minute adjustments to v:
    if (v>0).sum():
	v *= -1 # to ensure we have consistent evectors
    
    P = numpy.zeros((len(v)))
    S = numpy.ones((len(v)))
    for i in range(len(v)):
	P[i] = (v**(i+1)).sum()
    
    # now transform into S:
    S[0] = P[0] # do first manually
    for i in range(1,len(v)):
	if i%2==0:
	    S[i] = (P[0:(i+1)] * numpy.append(S[0:(i)][::-1],1) * (numpy.array([-1])**(xrange(i+1)))).sum() # S[0:(i+1)][::-1]
	else:
	    S[i] = (P[0:(i+1)] * numpy.append(S[0:(i)][::-1],1) * (numpy.array([-1])**(xrange(i+1))*-1)).sum() # S[0:(i+1)][::-1]
	S[i] *= ((-1)**(i))/(i+1.)
    
    return S
    
    
## some checks
#(1./2)*(S[0]*P[0] - P[1])
#(1./4)*( S[2]*P[0] - S[1]*P[1] + S[0]*P[2] - P[3] )
#(1./3) * ( S[1]*P[0] - S[0]*P[1] + P[2] )

