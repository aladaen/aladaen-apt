import pandas as pd 
from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from math import log2
from matplotlib import pyplot
from numpy.random import randint
import time
import numpy
from scipy.stats import wasserstein_distance
import sys
import os
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
def density_estimation(Estimation_mode,data_freq_rows,data_values,output_path):
   # if(not any(data_freq_rows) ):
    #    print("no data")
     #   return None
    
    
    if Estimation_mode=='kernel':
        model = KernelDensity(bandwidth=2, kernel='exponential')
        print("kernel gaussian estimation")
        print(model)
        model.fit(data_freq_rows)
        # sample probabilities for a range of outcomes
    
        probabilities = model.score_samples(data_values)
        probabilities = exp(probabilities)
        # plot the histogram and pdf
    # pyplot.hist(data_, bins=29, density=True)
       
        
    elif Estimation_mode=='gmms':
        model = GaussianMixture(n_components=3,init_params='random')
        model.fit(data_freq_rows)
        probabilities=model.predict(data_freq_rows)
    
    return probabilities
    

#####
##IC
def Information_Content(probabilities_vector,output_path):
    IC=[-log2(p+1E-9) for p in probabilities_vector]
    pyplot.clf()
    pyplot.plot(probabilities_vector,IC, marker='.')
    pyplot.title("Probability vs Information Content")
    pyplot.xlabel("Probability")
    pyplot.ylabel("Information Content")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pyplot.savefig(output_path+'/IC_'+str(timestr)+'.pdf', bbox_inches='tight')

    return IC

def Entropy(probabilities_vector):
    H=sum([-log2(p+1E-9)*p for p in probabilities_vector])
    print("Entropy of the bundle: {}".format(H))
    return H

def kullbal_liebler_divergence(proba_real,proba_generated):
    kl=sum(proba_real[i]*log2((proba_real[i]+1E-9)/(proba_generated[i]+1E-9)) for i in range(len(proba_real)))
    return kl

def Jensen_Shannon_divergence(proba_real,proba_generated):
    m=0.5*(proba_real+proba_generated)
    js=0.5*kullbal_liebler_divergence(proba_real,m)+0.5*kullbal_liebler_divergence(proba_generated,m)
    return js

def wasserstein_divergence(proba_real,proba_generated):
    w=wasserstein_distance(proba_real,proba_generated)
    return w

           
def Cross_Entropy(proba_real,proba_generated):
    CE= -sum([proba_real[i]*log2(proba_generated[i]+1E-9) for i in range(len(proba_real))])
    return CE

def moving_average(a, n) :
    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n