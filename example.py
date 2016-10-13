### Example script applied to toy data
#
#
# the simulated data has a cyclic covariance structure, with changes every 100 observations.
#

import numpy, os, pandas
import cPickle as pickle
import pylab as plt
from pyLGE_02 import *


os.chdir('Data/')
Nets = pickle.load(open('ToyNetworks.p', 'rb'))


### run PCA embedding:
Lpca = LGE(Adj = Nets, EmbedMethod='PCA', k=1)
ResPCA = Lpca.embed()

plt.plot( ResPCA[1].mean(axis=1), label='In-sample embedding')
plt.plot( ResPCA[2].mean(axis=1), label='Out-of-sample embedding')
plt.legend()
plt.title('PCA embedding')
plt.xlabel('Time')

### run LDA embedding:
# start by building response vector which is required:
ClassVec = numpy.zeros((300,2))
tID = numpy.linspace(0,299, 300 )
ClassVec[ (tID < 100) | (tID > 200), 0] = 1
ClassVec[ (tID >= 100) & (tID <= 200), 1] = 1
ClassFull = numpy.tile( ClassVec.T, len(Nets)).T

# now run embedding:
Llda = LGE(Adj=Nets, EmbedMethod='LDA', classID=ClassFull) 
ResLDA = Llda.embed()

plt.plot( ResLDA[1].mean(axis=1), label='In-sample embedding')
plt.plot( ResLDA[2].mean(axis=1), label='Out-of-sample embedding')
plt.legend(loc='best')
plt.title('LDA embedding')
plt.xlabel('Time')






