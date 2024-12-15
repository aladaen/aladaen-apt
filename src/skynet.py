import sys
import os
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
from matplotlib.pyplot import step, xlim, ylim, show
from math import sqrt
from matplotlib import pyplot
from numpy.random import rand
import matplotlib.pyplot as plt
from numpy import hstack
from matplotlib import pyplot
import pydotplus
import keras
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import pydot
from numpy import zeros
from numpy import ones
import numpy as np
from pathlib import Path
from numpy.random import randn
keras.utils.vis_utils.pydot = pydot
from discriminator.fake import generate_fake_samples  as disc_generate_fake_samples
from discriminator.real import generate_real_samples as disc_generate_real_samples
import anomalydetection.rare_rules
import anomalydetection.csv_to_rcf
from discriminator.real import get_real_samples as get_real_samples
from generator.latent import generate_latent_points as lt_generate_latent_points
from generator.latent import generate_fake_samples as lt_generate_fake_samples
from keras.optimizers import Adam
import pandas as pd
from numpy.random import randint
import distribution
from random import choice	
from numpy import asarray
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
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
import argparse
parser = argparse.ArgumentParser(description='APT-GAN Anomaly Detection')
parser.add_argument('--input', '-i',
					help=' an input context file', required=True)
parser.add_argument('--groundtruth', '-g', default='dummy.csv',
					help='A ground truth file')
parser.add_argument('--sup', '-s',
					help=' support ',
					default='0.05', required=True)
parser.add_argument('--iterations', '-t',
					help=' iterations to train the GAN ',
					default='500', required=True)
args = parser.parse_args()
input_file = os.path.abspath(args.input)
ground_truth= os.path.abspath(args.groundtruth)
sup=args.sup


opt = Adam(lr=0.0002, beta_1=0.5)
global_acc_real=list()
global_acc_fake=list()
global_Entropy=list()
global_IC=list()
global_Divergence=list()
global_wasserstein=list()
timestr = time.strftime("%Y%m%d-%H%M%S")
run_config="CP_1S_PE"

output_path=CURR_DIR+"/output/"+run_config+"/run_"+str(timestr)
if not os.path.exists(output_path):
    os.makedirs(output_path)
#input_file="ProcessEvent.csv"
#ground_truth="cadets_pandex_merged.csv"

output_scores=output_path+"/scores.csv"
output_ranks=output_path+"/rankings.csv"
conf=100
violator_list=list()
ndcg_list=list()
    #
save_fig=True
dataframe = pd.read_csv(input_file)#, nrows=10000)
dataset = dataframe.values[:,1:]
freq_rows_real=dataset.sum(axis=0)
length_seq=dataset.shape[1]
row_count=dataset.shape[0] 
#tmp_row,_=get_real_samples(dataset)
mat=np.zeros((1, length_seq+1))
bundle=np.zeros((1, length_seq))
bundle_per_epoch=np.zeros((1, length_seq))
bundle_width=100


values = asarray([value for value in range(1, length_seq)])
values = values.reshape((len(values), 1))
Estimation_mode='kernel'#,'gmm') 
freq_rows_real = freq_rows_real.reshape((len(freq_rows_real), 1))
proba_real=distribution.density_estimation(Estimation_mode,freq_rows_real,values,output_path)
poisoning=pd.DataFrame()



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
#TODO: keep the figure of generated samples that has best accuracy(real and fake)
# Gans discriminator model
def define_discriminator(n_inputs=1):
	model = Sequential()
	model.add(Dense(30, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def define_discriminator2(n_inputs=1):
    model = Sequential()
    #LSTM(100, dropout=0.2, recurrent_dropout=0.2)
    model.add(Dense(50,  kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(30,  kernel_initializer='he_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(20,  kernel_initializer='he_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(15,  kernel_initializer='he_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(10,  kernel_initializer='he_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(5,  kernel_initializer='he_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
	# compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

	
# train the discriminator model
def train_discriminator(model, n_epochs, n_batch):
     half_batch = length_seq
     acc_real_vect=list()
     acc_fake_vect=list()
     for i in range(n_epochs):
          X_real, y_real = get_real_samples(dataset)
          model.train_on_batch(X_real[:, 1], y_real)
          X_fake, y_fake = disc_generate_fake_samples(half_batch)
          model.train_on_batch(X_fake[:, 1], y_fake)
          _, acc_real = model.evaluate(X_real[:, 1], y_real, verbose=0)
          _, acc_fake = model.evaluate(X_fake[:, 1], y_fake, verbose=0)
          print(i, acc_real, acc_fake)
          acc_real_vect.append(acc_real)
          acc_fake_vect.append(acc_fake)
          #
          #row=np.column_stack([X_real[:, 1][0:length_seq],y_real])
          tmp=X_real[:, 1][0:length_seq]
         # print("\n\n\n############ x real shape ########\n\n\n")
          #print(tmp.shape)
          tmp.shape=(length_seq,)
          row=np.append(tmp,np.array([2]))
          row.shape=(length_seq+1,)
          #print(row)
          #row=np.append(X_real[:, 1][0:length_seq],np.array([20]))
          global mat
          mat=np.row_stack([mat,row])
          #plt.imshow(mat, cmap='hot', interpolation='nearest')
          #plt.show()
          #
          tmp2=X_fake[:, 1][0:length_seq]
          tmp2.shape=(length_seq,)
          row2=np.append(tmp2,np.array([3]))
          row2.shape=(length_seq+1,)
         # print(row2)
          
          
          #row=np.append(X_fake[:, 1][0:length_seq],np.array([40]))
          #row=np.column_stack([X_fake[:, 1][0:length_seq],y_fake])
          global mat
          mat=np.row_stack([mat,row2])
          #plt.imshow(mat, cmap='hot', interpolation='nearest')
          #plt.show()
     return (acc_real_vect,acc_fake_vect)
 
# define the standalone generator model
def define_generator(latent_dim, n_outputs=1):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(10,  kernel_initializer='he_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(5,  kernel_initializer='he_uniform'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# train the composite model
def train_gan(gan_model, latent_dim, n_epochs, n_batch):
	# manually enumerate epochs
	for i in range(n_epochs):
		# prepare points in latent space as input for the generator
		x_gan = lt_generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)


# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n):
   # global bundle_per_epoch
	# prepare real samples
    #dataset.shape
    x_real, y_real = get_real_samples(dataset)
 	# evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real[:, 1], y_real, verbose=0)
	# prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	# evaluate discriminator on fake examples
    x_fake.shape=(length_seq,)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    
	# scatter plot real and fake data points
    #print("Plot Summarize")
    #print(x_real)
    #print(x_fake)
   # pyplot.plot(range(0,len(x_real)), x_real, color='red')
   # pyplot.show()
    #pyplot.plot(range(0,len(x_fake)), x_fake, color='blue')
    #pyplot.show()
    

    

    
    
    #tmp=x_fake[0:length_seq]
    #print(tmp.shape)
    #tmp.shape=(length_seq,)
    #print(tmp.shape)
#
    #global mat
    #mat=np.row_stack([mat,tmp])
#    pyplot.imshow(mat, cmap='hot', interpolation='nearest')
    #plt.imshow(mat, cmap='hot', interpolation='nearest')
    #plt.show()

  
    return(acc_real,acc_fake)

def generate_latent_points(latent_dim, n):
    	# generate points in the latent space
	x_input=''.join(choice('01') for _ in range(latent_dim * n))
	x_input=np.fromstring(x_input,'u1') - ord('0')
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input


# use the generator to generate n fake examples and plot the results
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
 # lt_generate_fake_samples(g_model, latent_dim, half_batch)
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
    
    
    
def density_estimation(Estimation_mode,data_freq_rows,data_values,output_path):
       # if(not any(data_freq_rows) ):
    #    print("no data")
     #   return None
    
    global timestr 
    if Estimation_mode=='kernel':
        model = KernelDensity(bandwidth=2, kernel='gaussian')
        
        model.fit(data_freq_rows)
        # sample probabilities for a range of outcomes
    
        probabilities = model.score_samples(data_values)
        probabilities = exp(probabilities)
     #  pyplot.show()
        
    elif Estimation_mode=='gmms':
        model = GaussianMixture(n_components=3,init_params='random')
        model.fit(data_freq_rows)
        probabilities=model.predict(data_freq_rows)
           
    return probabilities
    
 
 
##IC
def Information_Content(probabilities_vector,output_path):
    IC=[-log2(p+1E-9) for p in probabilities_vector]

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
# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs, n_batch, n_eval):
    #x_real, y_real = disc_generate_real_samples( int(n_batch / 2))
	# determine half the size of one batch, for updating the discriminator
 #train(generator_model, discriminator_model, gan_model, latent_dim, n_epochs=n_iter, n_batch=by_batch,n_eval=length_seq)
    half_batch = length_seq
   # bundle_per_epoch=np.zeros((1, length_seq))
    global bundle
    Max_Div=1.0
    global poisoning
    poisoning_dataset = poisoning.values[:,1:]
    freq_rows_poisoning=poisoning_dataset.sum(axis=0)
    freq_rows_poisoning = freq_rows_poisoning.reshape((len(freq_rows_poisoning), 1))
    proba_poisoning=distribution.density_estimation(Estimation_mode,freq_rows_poisoning,values,output_path)
  
	# manually enumerate epochs
    for i in range(n_epochs):
        
        print("===================== epoch {}".format(i))
        object_list=list()
        #global bundle_per_epoch
        bundle_per_epoch_=np.zeros((1, length_seq))
        for j in range(0,bundle_width):
            # prepare real samples
            x_real, y_real = get_real_samples(dataset)
            # prepare fake examples
            x_fake, y_fake = lt_generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator
            #print("trining the disc on real")
            #print(x_real[:, 1].shape)
            #print(y_real.shape)
            
            d_model.train_on_batch(x_real[:, 1], y_real)
            #print("training the disc on fake")
            #print(x_fake.shape)
            #print(y_fake.shape)
            x_fake.shape=(length_seq,)
            #print(x_fake.shape)
            #print(y_fake.shape)
            d_model.train_on_batch(x_fake, y_fake)
            # prepare points in latent space as input for the generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            gan_model.train_on_batch(x_gan, y_gan)
            
          #  if (j+1) % 10 == 0:
           #     _,_=summarize_performance(j, g_model, d_model, latent_dim,n_eval)

		# evaluate the model every n_eval epochs
        if (i+1) % 10 == 0:
            #j=0
            accur_real,accur_fake=summarize_performance(i, g_model, d_model, latent_dim,n_eval)
            global_acc_real.append(accur_real)
            global_acc_fake.append(accur_fake)
            
            for k in range (1,bundle_width*2):
                x_fake, y_fake = lt_generate_fake_samples(g_model, latent_dim, half_batch)
                tmp=x_fake[0:length_seq]
                #print(tmp.shape)
                tmp.shape=(length_seq,)
                #print(tmp.shape)
                row=np.append(tmp,np.array([3]))
                #tmp.shape=(length_seq+1,)
                row.shape=(length_seq+1,)
                print("----added attack row")
                print(row)
                ############################################
                global mat
                mat=np.row_stack([mat,row])
                #plt.imshow(mat, cmap='hot', interpolation='nearest')
                #plt.show()
                ################################
                row_bndle=x_fake[0:length_seq]
                row_bndle.shape=(length_seq,)
                object_list.append("object_"+str(k))
                #if(accur_real !=0 and accur_fake != 0):
                bundle_per_epoch_=np.row_stack([bundle_per_epoch_,row_bndle])
                print(bundle_per_epoch_)


        
            print("==== Now analyzing bundle number : {}".format(i))  
      
            print("generating the distribution in epoch : {}".format(i))        
            #delete first row
            bundle_per_epoch_= np.delete(bundle_per_epoch_, (0), axis=0)
            freq_rows_bundle=bundle_per_epoch_.sum(axis = 0)
            freq_rows_bundle = freq_rows_bundle.reshape((len(freq_rows_bundle), 1))
            #print("*** vector of frequencies \n")
            #freq_rows_bundle.shape
            #print(freq_rows_bundle)
            #if(not any(freq_rows_bundle) ):
             #   next
            #global Estimation_mode
            #global values
            proba_generated=density_estimation(Estimation_mode,freq_rows_bundle,values,output_path)
            

        
        
           # if(not any(proba_generated)):
            #    next
            print("=== Information Content of the bundle in epoch {} ".format(i))
           # print(proba_generated)
            #print(proba_generated.shape)  
            IC=Information_Content(proba_generated,output_path)
            #global_IC.append(np.mean(IC))
            H=Entropy(proba_generated)
            #global_Entropy.append(H)
            print("=== Entropy of the bundle in epoch {} is {}".format(i,H))
            #global proba_real
#            from scipy.stats import wasserstein_distance
            Div=sqrt(Jensen_Shannon_divergence(proba_poisoning,proba_generated))
            print("=== Jensen Shannon / kullback lliebler divergence  in epoch {} is {}".format(i,Div))
            was=wasserstein_divergence(proba_poisoning,proba_generated)
            print("=== Waserstein divergence  in epoch {} is {}".format(i,was))
            
         
            #global_Divergence.append(Div)
            #if kl < 0.5 add global bundle to bundle
        #  global bundle
        # 
            if(Div>0.6):#was>=0.9):
                print("Divergence >=0.5 the bundle is different")
            else:
            #if(Div<=Max_Div):
                   
                if (np.mean(IC)>0):
                    #Max_Div=Div
                    print("the bundle is close and ==> append")
                    
                    #]==================================================================================================
                    bundle_df=pd.DataFrame(bundle_per_epoch_)
                    bundle_per_epoch_.shape
                    #df.insert(loc, column, value)
                    bundle_df.insert(0,"Object_ID",object_list)
                    del bundle_df.index.name
                    #bundle_df["Object_ID"]=object_list #df["D"] = m[df.A]
                    
                    bundle_df.columns=dataframe.columns
                    bundle_df=bundle_df.astype(int, errors='ignore')
                    bundle_df = pd.concat([poisoning,bundle_df],ignore_index=True)
                    bundle_input_file=output_path+"/Bundle_Epoch_"+str(i)+".csv"
                    bundle_df.to_csv(bundle_input_file,index=False , header=True)
                    bundle_output_scores=output_path+"/Scores_Epoch_"+str(i)+".csv"
                    bundle_output_ranks=output_path+"/Ranks_Epoch_"+str(i)+".csv"
                    sup=0.05
                    rcf_file=os.path.splitext(bundle_input_file)[0]+".rcf"
                    anomalydetection.csv_to_rcf.get_rcf(bundle_input_file,rcf_file)
                    try:
                        bundle_ndcg=anomalydetection.rare_rules.extract_rare_rules( bundle_input_file,bundle_output_scores,bundle_output_ranks,True,ground_truth,sup,conf )
                        ndcg_list.append(float(bundle_ndcg[0]))
                    except:
                        try:
                            sup=5
                            bundle_ndcg=anomalydetection.rare_rules.extract_rare_rules( bundle_input_file,bundle_output_scores,bundle_output_ranks,True,ground_truth,sup,conf )
                            ndcg_list.append(float(bundle_ndcg[0]))
                        except:
                            
                            try:
                                sup=20
                                bundle_ndcg=anomalydetection.rare_rules.extract_rare_rules( bundle_input_file,bundle_output_scores,bundle_output_ranks,True,ground_truth,sup,conf )
                                ndcg_list.append(float(bundle_ndcg[0]))
                                
                            except:
                                print("An exception occurred") 
                                bundle_ndcg=0.0
                                ndcg_list.append(bundle_ndcg)
                    
                    
                    #len(dataframe['Object_ID'])
                    bundle_violators=pd.read_csv(bundle_output_scores) 
                    #violator_list= list(violators['x'])
                   # len(violator_list)
                    violator_list.append(len(bundle_violators))
                    #poisoning= dataframe[dataframe['Object_ID'].isin(violator_list)]
                    #len(poisoning['Object_ID'])
                  #  poisoning_dataset = poisoning.values[:,1:]
                   # freq_rows_poisoning=poisoning_dataset.sum(axis=0)
                    #freq_rows_poisoning = freq_rows_poisoning.reshape((len(freq_rows_poisoning), 1))
                    #proba_poisoning=density_estimation(Estimation_mode,freq_rows_poisoning,values,output_path)
    
    
                    
            
                    ##====================================================================================
                    global_IC.append(np.mean(IC))
                    global_Entropy.append(H)
                    global_Divergence.append(Div)
                    global_wasserstein.append(was)

 
    
    global_IC_hat=  distribution.moving_average(global_IC, 10) # window size 51, polynomial order 3
    global_Entropy_hat = distribution.moving_average(global_Entropy, 10)# window size 51, polynomial order 3
    global_Divergence_hat = distribution.moving_average( global_Divergence, 10) # window size 51, polynomial order 3
    global_wasserstein_hat= distribution.moving_average(global_wasserstein, 10) # window size 51, polynomial order 3

    ndcg_hat= distribution.moving_average(ndcg_list, 10) # window size 51, polynomial order 3
    print("now printing ndcg scores")
    print(ndcg_hat)
    pyplot.clf()
    pyplot.plot(range(0,len(ndcg_hat)), ndcg_hat, color='red')
    pyplot.title("Smoothing nDCG scores with poisoning attacks")
    pyplot.xlabel("Epochs (# bundle)")
    pyplot.ylabel("nDCG")
    pyplot.savefig(output_path+'/Smoothing_Ndcg_'+str(timestr)+'.pdf', bbox_inches='tight')
    pyplot.show()
    


    

    
    pyplot.clf()
    pyplot.plot(range(0,len(global_IC_hat)), global_IC_hat, color='blue',label='IC')
   # pyplot.plot(range(0,len(global_Entropy_hat)), global_Entropy_hat, color='green',label='H')
   # pyplot.plot(range(0,len(global_Divergence_hat)), global_Divergence_hat, color='red',label='JS')
    #pyplot.plot(range(0,len(global_Divergence_hat)), global_Divergence_hat, color='green')
    #pyplot.plot(range(0,len(global_wasserstein_hat)), global_wasserstein_hat, color='purple',label='WAS')
    pyplot.legend(loc="upper left")
    pyplot.title("Smoothed metrics of the bundles")
    pyplot.xlabel("Epochs (# bundle)")
    pyplot.ylabel("Metrics")
    
    pyplot.savefig(output_path+'/Smoothed_IC_'+str(timestr)+'.pdf', bbox_inches='tight')
    pyplot.show()

#                  



   
if __name__ == "__main__":
        
    ###first step: running the anomaly detection to identify the violator list=TP+FP
    ###extracting the true positives

    try:
        poison_ndcg=anomalydetection.rare_rules.extract_rare_rules( input_file,output_scores,output_ranks,True,ground_truth,sup,conf )
        ndcg_list.append(float(poison_ndcg[0]))
    except:
        print("An exception occurred") 
        poison_ndcg=0.0
        ndcg_list.append(poison_ndcg)
                        
         
    
    
    df_viol=pd.read_csv(output_scores) 
    violators= list(df_viol['x'])
    
    violator_list.append(len(violators))
    poisoning= dataframe[dataframe['Object_ID'].isin(violators)]
    
    poisoning_dataset = poisoning.values[:,1:]
    freq_rows_poisoning=poisoning_dataset.sum(axis=0)
    freq_rows_poisoning = freq_rows_poisoning.reshape((len(freq_rows_poisoning), 1))
    proba_poisoning=distribution.density_estimation(Estimation_mode,freq_rows_poisoning,values,output_path)
        
           
    dataReal, classesReal =get_real_samples(dataset)
    tmp=dataReal[:, 1][0:length_seq]
    tmp.shape=(length_seq,)
    
    row=np.append(tmp,np.array([2]))
    #global mat
    row.shape=(length_seq+1,)
    mat=np.row_stack([mat,row])
    
            
    # generate fake samples
    dataFake, classesFake = disc_generate_fake_samples(length_seq)
    # plot samples
    tmp=dataFake[:, 1][0:length_seq]
    
    tmp.shape=(length_seq,)

    row=np.append(tmp,np.array([3]))
    #tmp.shape=(length_seq+1,)
    row.shape=(length_seq+1,)
    #print(row)
    mat=np.row_stack([mat,row])




    # define the discriminator model
    discriminator_model = define_discriminator2()
    # summarize the model
    discriminator_model.summary()
    # plot the model
    
    # fit the model
    n_iter=1000 #int(2*row_count/3)
    by_batch=length_seq
    real_acc,fake_acc=train_discriminator(discriminator_model,n_iter,by_batch)
    latent_dim = int(3*length_seq/4)
    generator_model = define_generator(latent_dim)
    # summarize the model
    generator_model.summary()
    gen_data,_=lt_generate_fake_samples(generator_model, latent_dim, length_seq)
    tmp=gen_data[0:length_seq]
    
    tmp.shape=(length_seq,)
    row=np.append(tmp,np.array([3]))
    #tmp.shape=(length_seq+1,)
    row.shape=(length_seq+1,)
    #print(row)
    mat=np.row_stack([mat,row])
       
    
#    pyplot.imshow(mat, cmap='hot', interpolation='nearest')
    
    gan_model = define_gan(generator_model, discriminator_model)
    gan_model.summary()
    # plot gan model
    
    
    
    train(generator_model, discriminator_model, gan_model, latent_dim, 
          n_epochs=int(args.iterations),#n_iter, 
          n_batch=by_batch,n_eval=length_seq)