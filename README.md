### Python implementation of linear graph embedding methods
#
# 
# 
#

This module contains a python implementation of linear graph embedding methods based on PCA and LDA.
These embedding methods are implemented via a the `SINGLE` class which has the following methods:

1. `getLaplacianMatrix`: builds Laplacian matrices based on adjacency matrices
2. `embed`: runs linear graph embedding method (either PCA or LDA driven, as specified at object initialization)

#### Example:
We provide toy simulated data corresponding to 10 subjects in the `Sample Data` folder. 

Below is a short script, for the full script please see `example.py` file 

```
import numpy, os, pandas
import cPickle as pickle
from pyLGE_02 import *

os.chdir('Data/')
Nets = pickle.load(open('ToyNetworks.p', 'rb'))

### run PCA embedding:
Lpca = LGE(Adj = Nets, EmbedMethod='PCA', k=1)
ResPCA = Lpca.embed()

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
```

![alt text](https://raw.githubusercontent.com/piomonti/pyLGE/blob/master/ExampleFig.png "")
