import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import time
import numba
import os
from joblib import Parallel, delayed
import multiprocessing
import pickle
import joblib
import subprocess
import platform
from tqdm import tqdm

class WSmethod:
	def __init__(self, num_node, num_edge, p1, p2, num_sensor, num_motor, inter_net_delay, condition = "random"):
		inter_net_delay = int(inter_net_delay)
		network1 = nx.watts_strogatz_graph(num_node, num_edge, p1)
		network2 = nx.watts_strogatz_graph(num_node, num_edge, p2)

		temp = np.zeros(num_node**2).reshape(num_node,num_node)

		if condition == "random":
			sensor = np.empty(num_sensor)
			motor  = np.empty(num_motor)
			arr  = np.array(range(num_node))
			LIST = arr.tolist()
			ARR  = np.random.choice(LIST, 19, replace = False)
			for i in xrange(sensor.shape[0]):
				sensor[i] = ARR[i]
			for i in xrange(sensor.shape[0], sensor.shape[0] + motor.shape[0]):
				motor[i - sensor.shape[0]]  = ARR[i]
			self.sensor = np.array(sensor, np.int32)
			self.motor  = np.array(motor,  np.int32)
			for i, j in itertools.product( xrange(self.sensor.shape[0]), xrange(self.motor.shape[0]) ):
				temp[self.sensor[i], self.motor[j]] = 1
		elif condition == "segregated":
			for i, j in itertools.product( xrange(num_sensor), xrange(num_motor) ):
				temp[i,j + num_node/2 + 1] = 1


		self.network1 = network1
		self.network2 = network2
		adj_mat1     = nx.to_numpy_matrix(self.network1)
		adj_mat2     = nx.to_numpy_matrix(self.network2)
		weight_set1  = np.random.rand(num_node**2).reshape(num_node,num_node)
		weight_set2  = np.random.rand(num_node**2).reshape(num_node,num_node)
		weight_mat1  = np.multiply(adj_mat1, weight_set1)
		weight_mat2  = np.multiply(adj_mat2, weight_set2)
		delay_set1   = 20*np.random.rand(num_node**2).reshape(num_node,num_node)+10
		delay_set2   = 20*np.random.rand(num_node**2).reshape(num_node,num_node)+10
		delay_mat1   = np.multiply(adj_mat1, delay_set1 ).astype(np.int32)
		delay_mat2   = np.multiply(adj_mat2, delay_set2 ).astype(np.int32)

		weight_mat1 = (weight_mat1 + weight_mat1.T) / 2 # to make symmertry matrix
		weight_mat2 = (weight_mat2 + weight_mat2.T) / 2 # to make symmertry matrix
		delay_mat1  = (delay_mat1  + delay_mat1.T ) / 2 # to make symmertry matrix
		delay_mat2  = (delay_mat2  + delay_mat2.T ) / 2 # to make symmertry matrix

		self.adj_mat1    = adj_mat1
		self.adj_mat2    = adj_mat2
		self.weight_mat1 = weight_mat1
		self.weight_mat2 = weight_mat2
		self.delay_mat1  = delay_mat1
		self.delay_mat2  = delay_mat2

		avg_weight1 = np.mean(self.weight_mat1)
		avg_weight2 = np.mean(self.weight_mat2)
		avg_weight  = (avg_weight1+avg_weight2)/2

		A1 = np.concatenate( ( self.adj_mat1,  temp ),                 axis = 0 )
		A2 = np.concatenate( ( temp,  self.adj_mat2 ),                 axis = 0 )
		W1 = np.concatenate( ( self.weight_mat1, temp*avg_weight ),    axis = 0 )
		W2 = np.concatenate( ( temp*avg_weight, self.weight_mat2 ),    axis = 0 )
		D1 = np.concatenate( ( self.delay_mat1, temp*inter_net_delay), axis = 0 )
		D2 = np.concatenate( ( temp*inter_net_delay, self.delay_mat2), axis = 0 )

		self.adj_mat    = np.concatenate((A1,A2), axis = 1) # hyper adjacent matrix
		self.weight_mat = np.concatenate((W1,W2), axis = 1)
		self.delay_mat  = np.concatenate((D1,D2), axis = 1)
		self.hyper_mat  = temp
		self.avg_weight = avg_weight
		self.inter_net_delay = inter_net_delay









































