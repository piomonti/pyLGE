### Example script applied to toy data
#
#
# the simulated data has a cyclic covariance structure, with changes every 100 observations.
#

import numpy, os, pandas
import cPickle as pickle
import pylab as plt
import matplotlib.collections as collections
from pyLGE_02 import *


os.chdir('Data/')
Nets = pickle.load(open('ToyNetworks.p', 'rb'))

# prepare image:
f, (ax1, ax2) = plt.subplots(1, 2)
t = numpy.linspace(1, 300,300)

### run PCA embedding:
Lpca = LGE(Adj = Nets, EmbedMethod='PCA', k=1)
ResPCA = Lpca.embed()

ax1.plot( ResPCA[1].mean(axis=1), label='In-sample embedding', linewidth=2.5)
ax1.plot( ResPCA[2].mean(axis=1), label='Out-of-sample embedding', linewidth=2.5)
ax1.legend()
ax1.set_title('PCA embedding', fontsize=25)
ax1.set_xlabel('Time')
collection = collections.BrokenBarHCollection.span_where(t, ymin=-1, ymax=1, where=((t<100) | (t>200)), facecolor='blue', alpha=0.25, label='0 back')
ax1.add_collection(collection)

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

ax2.plot( ResLDA[1].mean(axis=1), label='In-sample embedding', linewidth=2.5)
ax2.plot( ResLDA[2].mean(axis=1), label='Out-of-sample embedding', linewidth=2.5)
ax2.legend(loc='best')
ax2.set_title('LDA embedding', fontsize=25)
ax2.set_xlabel('Time')
collection = collections.BrokenBarHCollection.span_where(t, ymin=-1, ymax=1, where=((t<100) | (t>200)), facecolor='blue', alpha=0.25, label='0 back')
ax2.add_collection(collection)





