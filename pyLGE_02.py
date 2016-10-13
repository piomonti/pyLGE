### python class implementing linear graph methods described in Monti et al (2016)
#
#
# We implement linear graph embeddings based on PCA and LDA
#
#
#

import numpy, os
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from SpectralFunctions import *
from CrossValidateSLDA import *
from OneVsRestLDAClassifiers import *

class LGE():
    """Class for Linear Graph Embeddings (LGE)
    
    Has the following methods:
    	- __init__: initialize LGE object
    	- embed: method to embed time-varying functional connectivity networks
    	-

    """

    def __init__(self, Adj, EmbedMethod='PCA', k=None, classID=None, lamRange=None, stabPercent=None):
    	"""
		INPUT:
			- Adj: list of lists.
				Each list contains the time-varying adjacency/connectivity matrices for a given subject. 
				Each entry should be a p by p symmetric matrix
			- EmbedMethod: should be either 'PCA' or 'LDA'. Indicates which embedding method to employ. Defaults to 'PCA'
			- k: number of principal components to calculate if EmbedMethod=='PCA'
			- classID: matrix indicating classes for LDA driven embedding. One column per each class
			- lamRange: range of candidate regularization parameters employed in CV of sparse LDA model (done on a subject by subject basis)
			- stabPercent: percentage of time an edge must be present across subjects to pass the screening procedure

    	"""
    	assert len(set([x.shape for x in Adj[0]]))==1, 'adjacency matrices of varying dimension provided!' # we only check the first matrix. Could be problems further down...
    	self.Adj = Adj
    	self.EmbedMethod = EmbedMethod
    	self.Laplacian = None # will store Laplacian matrices here 
    	self.p = self.Adj[0][0].shape[0]
    	self.k = k # number of PCs to compute#
    	self.n = len(self.Adj[0]) # number of networks per subject, assumed to be the same across all subjects!
    	self.classID = classID
    	self.lamRange = lamRange
    	self.stabPercent = stabPercent


	
    def getLaplacianMatrix(self):
    	"""
    	Build L matrix

    	"""
    	subLens = [len(x) for x in self.Adj] # get number of networks per subject
    	Laplacian = numpy.zeros(( numpy.sum(subLens), int(self.p*(self.p-1)*.5)  ))

    	# now we go through and populate each entry:
    	counter = 0
    	for subNet in self.Adj:
    		for x in range(len(subNet)):
    			Laplacian[counter,:] = getL(subNet[x], Binary=False)[numpy.triu_indices(self.p, k=1)]
    			counter += 1

    	return Laplacian



    def embed(self):
		"""
		Run linear graph embedding

		Embedding method specified at initialization will be used

		OUTPUT:
			- principal components or linear discriminant (depending on embedding method)
			- in-sample embedding (i.e., embedding applied to training data)
			- out-of-sample embedding (i.e., embedding applied to test data)

		"""

		if self.EmbedMethod=='PCA':
			print 'Running PCA-driven embedding'
			# check if Laplacians have been calculated:
			if self.Laplacian == None:
				self.Laplacian = self.getLaplacianMatrix()

			if self.k==None:
				print 'By default taking leading 2 principal components'
				self.k=2

			# we implement 50% training data and 50% test data by default
			self.Ltrain = self.Laplacian[ :int(self.Laplacian.shape[0]/2), :]
			self.Ltest = self.Laplacian[ int(self.Laplacian.shape[0]/2):, :]

			# run PCA embedding:
			pca = PCA(n_components=self.k)
			PCs = pca.fit(self.Ltrain)
			
			return PCs.components_, PCs.transform(self.Ltrain).reshape(( self.n, -1, self.k), order='F'), PCs.transform(self.Ltest).reshape((self.n, -1, self.k), order='F')

		if self.EmbedMethod=='LDA':
			print 'Running LDA-driven embedding'
			# check if Laplacians have been calculated:
			if self.Laplacian == None:
				self.Laplacian = self.getLaplacianMatrix()
			
			if self.lamRange==None:
				self.lamRange = numpy.linspace(.01, .1, 10) # default

			if self.stabPercent==None:
				self.stabPercent = .5 # default

			# perform variable screening by stability selection -  note we assume the same number of observations for each subject
			edgeCounts = numpy.zeros((self.Laplacian.shape[1], ))
			for i in range(len(self.Adj)):
				# fit sparse LDA model for each subject - select reg parameter via CV:
				Res = CVsparseLDA(pred = self.Laplacian[ (self.n * i):(self.n * (i+1)), : ], resp=self.classID[(self.n * i):(self.n * (i+1)), :], l1range=self.lamRange)
				l1Val = self.lamRange[Res[0].mean(axis=1).argmax()] # select score which maximizes AUC on unseen data:
				# finally fit regularized LDA model for this subject:
				mod = SparseLDA(Y = self.classID[(self.n * i):(self.n * (i+1)), :], X = self.Laplacian[ (self.n * i):(self.n * (i+1)), : ], alpha=l1Val, l1_ratio=1)
				edgeCounts += (mod.reshape((-1,)) !=0)

			screenIDs = ((edgeCounts/len(self.Adj)) >= self.stabPercent)
			print str(screenIDs.sum()) + ' edges selected via stability selection'

			# we implement 50% training data and 50% test data by default
			self.Ltrain = self.Laplacian[ :int(self.Laplacian.shape[0]/2), screenIDs]
			self.Ltest = self.Laplacian[ int(self.Laplacian.shape[0]/2):, screenIDs]
			SLDA = numpy.zeros(( self.Laplacian.shape[1], 1))
			SLDA[screenIDs,] = SparseLDA(Y=self.classID[:int(self.Laplacian.shape[0]/2)], X=self.Ltrain, alpha=.005, l1_ratio=1) # l1_ratio is ratio of l1 to l2 reg, which we always set to 1 for sparsity

			return SLDA, numpy.dot(self.Ltrain, SLDA[screenIDs,]).reshape((self.n, -1), order='F'), numpy.dot(self.Ltest, SLDA[screenIDs,]).reshape((self.n, -1), order='F')
	


