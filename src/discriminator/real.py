from numpy.random import rand
from numpy import hstack
from numpy import zeros
from numpy import ones
import numpy as np
from random import choice	
import pandas as pd
from numpy.random import randn
from numpy.random import randint

def generate_real_samples(n):
    # generate inputs in [-0.5, 0.5]
    X1=np.array(range(0,n))
    #X1=np.fromstring(X1,'u1') - ord('0')
    #X1=np.array(X1)
	# 
	#  
    X2=''.join(choice('01') for _ in range(n))
    X2=np.fromstring(X2,'u1') - ord('0')
	# 
	# stack arrays 
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
	# generate class labels
    y = ones((n, 1))
    return X, y



def get_real_samples(dataset):
    
    # load the dataset

    #
    	# choose random instances
    ix = randint(0, dataset.shape[0], 1)
	# retrieve selected images
    X2 = dataset[ix].astype(str).astype(int)
    
	# generate 'real' class labels (1)
    n=dataset.shape[1]

    y = ones((n, 1))
    #return X, y

    # generate inputs in [-0.5, 0.5]
    X1=np.array(range(0,n))
    #X1=np.fromstring(X1,'u1') - ord('0')
    #X1=np.array(X1)
	# 
	#  
  #  X2=''.join(choice('01') for _ in range(n))
   # X2=np.fromstring(X2,'u1') - ord('0')
	# 
	# stack arrays 
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
	# generate class labels
    y = ones((n, 1))
    return X, y