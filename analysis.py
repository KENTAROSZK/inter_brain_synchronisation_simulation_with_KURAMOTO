import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import itertools
from scipy import signal
import seaborn as sns
import pandas as pd
from scipy import fftpack
from scipy import signal
import random
import numba
from numba import prange
from joblib import Parallel, delayed
import numpy as np



def bandpassfilter(x,f_low,f_high):
	print( "signal_analysis bandpassfilter")
	num_of_samples = x.shape[1]
	sample_rate = 1000 # sampling rate fs
	sampleF = float(1)/sample_rate
	nyq   = 1/(sampleF*2) # Nyquist frequency
	fe1  = f_low /nyq     # cutt off freq (lower)
	fe2  = f_high/nyq     # cutt off freq (higher)
	temp = int(num_of_samples)/5
	if temp % 2 == 0:
		temp -=1
	numtaps = temp

	# setting bandpass filter
	# bandpass = signal.firwin(numtaps, [fe1,fe2], pass_zero=False)
	bandpass = signal.firwin(numtaps, cutoff=[fe1,fe2], pass_zero=False)

	# filtering datas
	filtered_data = signal.lfilter(bandpass,1, x, axis=1)

	return filtered_data



def before_complex_number(x, num_node, time):
	data = np.empty( (num_node*num_node, time) , dtype=np.float32)
	for i,ind in itertools.product( xrange(num_node), xrange(num_node) ):
		data[i*num_node+ind] = x[i] -x[ind+num_node]

	return data



def hilbert_transform(x): # equivalent to phase
	print( "signal_analysis bandpassfilter")
	phase = np.angle(signal.hilbert(x,axis = 1),deg =False) # radian, unnecessary to unwrap
	return phase


def plv(x, num_slice = 10):
	print("signal_analysis plv")
	# X is data array , the shape of it is 
	# node num * time
	# num_slice is how many times you wanna slice data
	# default is 10
	time       = x.shape[1]
	num_node   = x.shape[0]/2
	windowsize = time/num_slice

	data = np.exp(1j*before_complex_number(x, num_node, time))

	each_point_plv = np.empty( (num_node*num_node, num_slice) ).astype(np.float16) # initialise ndarray to contain plv time series

	for k in xrange(num_slice):
		DT                  = data[:, k*windowsize:(k+1)*windowsize]
		temp                = np.sum(DT, axis=1)
		each_point_plv[:,k] = np.abs(temp)/windowsize

	return each_point_plv


def plot_plv(x, name = ""):
	num_node = int(np.sqrt(x.shape[0]))
	plt.close()

	os.chdir(os.getcwd())


	time = x.shape[1]
	arr = np.empty( (x.shape[0],1) )
	plv_matrix = np.empty( (num_node,num_node) )
	INDEX  = []
	COLUMN = []
	for i in range(num_node):
		INDEX.append( "NET1 No."+str(i+1))
		COLUMN.append("NET2 No."+str(i+1))
	for t in range(time):
		arr[:,0] = x[:,t]
		for i,j in itertools.product( range(num_node), range(num_node) ):
			#print "i:" + str(i) + "// j:" + str(j)
			plv_matrix[i,j] = arr[i*num_node+j]
		df = pd.DataFrame(data = plv_matrix,index = INDEX, columns = COLUMN)
		plt.figure()
		#plt.rcParams["font.size"] = 10
		sns.heatmap(df, vmax=1.0, vmin=-0.0, center=0.35, cmap = "jet")
		plt.title("PLV (time point:" + str(t+1) + ")" + name)
		plt.tight_layout()
		plt.savefig(name + "_PLV_time_point_" + str(t+1) + ".png")
		plt.close()


