from numpy.random import rand
from numpy import hstack
from numpy import zeros
from numpy import ones
import numpy as np
from numpy.random import randn

from matplotlib import pyplot
import pydotplus
import pydot
from random import choice	


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	#x_input = randn(latent_dim * n)
	x_input=''.join(choice('01') for _ in range(latent_dim * n))
	x_input=np.fromstring(x_input,'u1') - ord('0')
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input


# use the generator to generate n fake examples and plot the results
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)
    if(np.mean(X)<0.1):
        X[X>np.mean(X)]=1
        X[X<=np.mean(X)]=0
    else:
        X[X>0.5]=1
        X[X<=0.5]=0
    y = zeros((n, 1))
    return X, y
    