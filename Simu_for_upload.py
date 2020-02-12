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

from WSmethod import WSmethod
from Pic_Net import Take_pic
import analysis


class params:
	def __init__(self, num_node):
		"""
		natural_freq : natural frequency for KURAMOTO oscillators
		simu_time    : total simulation length (second)
		shift        : this is a parameter for simulation. must be longer than conduction delay (e.g., in this example, the longest delay is 30msec, so shift must be larger than 30.)
		num_node     : the number of the nodes for each network
		num_edge     : the number of the edges for each node
		"""
		self.natural_freq  = [31.0, 48.0, "gamma"]
		self.simu_time     = 10     # sec
		self.sampling_rate = 1000   # Hz
		self.shift         = 150    # msec
		self.dt            = float(1)/self.sampling_rate
		self.num_node      = num_node




class initialise_params:
	def __init__(self, params):
		"""
		init_phase : initial phase 0~2pi
		omega      : natural frequency for each oscillator
		phase      : phase for each oscillators at every time points. 

		phase = [(the size of each network * 2), (simu_time*sampling_rate)]
		the first half of row is network1's phase, and the second half is network2's phase.

		for example, when the number of node for each network is 100, simu_time is 10 (sec), and sampling_rate is 1000 (Hz),
		phase size is [(2*100) , (10*1000)]
		phase[:100,:] <-- network1's phase through simulation
		phase[100:,:] <-- network2's phase through simulation
		"""
		mean_freq = (params.natural_freq[0] + params.natural_freq[1])/2
		std_freq  = params.natural_freq[1] - mean_freq

		self.init_phase = 2 * np.pi * np.random.rand(params.num_node*2)                        # initial state of phase: 0~2pi
		self.omega      = 2 * np.pi * np.random.normal(mean_freq, std_freq, params.num_node*2) # natural frequency
		self.phase      = np.empty(params.num_node*2*(params.simu_time*params.sampling_rate+params.shift), dtype='float32').reshape(params.num_node*2,(params.simu_time*params.sampling_rate+params.shift))

		for t in range(params.shift):
			self.phase[:,t] = self.init_phase[:] - t * self.omega[:]
		self.phase = np.mod(self.phase, 2*np.pi)


	def combine_matrices(self, mat1, mat2, mat12, mat21):
		"""
		mat1  : network1's matrix
		mat2  : network2's matrix
		mat12 : network1 -> network2's hyper matrix, which means network1 effects on network2
		mat21 : network2 -> network1's hyper matrix, which means network2 effects on network1
		"""
		A1 = np.concatenate( ( mat1,  mat12 ), axis = 0 )
		A2 = np.concatenate( ( mat21, mat2  ), axis = 0 )
		A  = np.concatenate( (A1,     A2),     axis = 1 )

		return A



def main(params_4_simu):
	phase           = params_4_simu["phase"]
	adj_mat         = params_4_simu["adj_mat"]
	weight_mat      = params_4_simu["weight_mat"]
	delay_mat       = params_4_simu["delay_mat"]
	omega           = params_4_simu["omega"]
	shift           = params_4_simu["shift"]
	dt              = params_4_simu["dt"]
	total_simu_time = params_4_simu["simulation_time"]
	Cintra          = params_4_simu["Cintra"]
	Cinter          = params_4_simu["Cinter"]

	return Kuramoto(phase, adj_mat, weight_mat, delay_mat, omega, total_simu_time, shift, dt, Cintra, Cinter)



@numba.jit # speeding up the Kuramoto computation by numba
def Kuramoto(phase, adj_mat, weight_mat, delay_mat, omega, total_simu_time, shift, dt, Cintra, Cinter):
	num_node   = phase.shape[0]
	intra_size = int(num_node / 2)

	for t in xrange(total_simu_time):
		T = t + shift - 1
		for i in xrange(num_node):
			intra_SUM = 0.0
			inter_SUM = 0.0
			chk_net = i // intra_size # if chk_net == 1 -> net2, otherwise net1
			for j in xrange(intra_size): ####### compute intra
				if chk_net == 0: # net1
					k = j
				else:            # net2
					k = (-1) * (j+1)
				intra_SUM += weight_mat[i,k] * np.sin(phase[k, T-int(delay_mat[i,k])] - phase[i,T])
			intra_SUM *= Cintra
			for j in xrange(intra_size, num_node): ####### compute inter
				if chk_net == 0: # net1
					k = j
				else:            # net2
					k = (-1) * (j+1)
				inter_SUM += weight_mat[i,k] * np.sin(phase[k, T-int(delay_mat[i,k])] - phase[i,T])
			inter_SUM *= Cinter

			SUM = omega[i] + intra_SUM + inter_SUM + np.random.randn()
			phase[i,T+1] = SUM*dt + phase[i,T]
			phase[i,T+1] = np.mod(phase[i,T+1], 2*np.pi)

	return phase[:,shift:]
	"""
	in order to cut the first shift time, return the "phase[:,shift:]"
	"""




if __name__ == "__main__":
	num_node   = 100      # num of nodes per each network
	num_edge   = 6
	num_sensor = int(10*num_node/90)
	num_motor  = int(8 *num_node/90)
	p1 = p2 = 0.1
	inter_net_delay = 10.0 # msec
	condition = "segregated"
	net = WSmethod(num_node, num_edge, p1, p2, num_sensor, num_motor, inter_net_delay, condition)

	Cintra = 100.0
	Cinter = 10.0

	pms        = params(net.adj_mat1.shape[0])
	init_pms   = initialise_params(pms)
	adj_mat    = init_pms.combine_matrices(net.adj_mat1, net.adj_mat2, net.hyper_mat, net.hyper_mat)
	weight_mat = init_pms.combine_matrices(net.weight_mat1, net.weight_mat2, net.avg_weight*net.hyper_mat, net.avg_weight*net.hyper_mat)
	delay_mat  = init_pms.combine_matrices(net.delay_mat1, net.delay_mat2, net.inter_net_delay*net.hyper_mat, net.inter_net_delay*net.hyper_mat)



	params_4_simu = {
	"phase"          :init_pms.phase,
	"adj_mat"        :adj_mat,
	"weight_mat"     :weight_mat,
	"delay_mat"      :delay_mat,
	"omega"          :init_pms.omega,
	"shift"          :pms.shift,
	"dt"             :pms.dt,
	"simulation_time":pms.simu_time*pms.sampling_rate,
	"Cintra"         :Cintra,
	"Cinter"         :Cinter
	}

	phase = main(params_4_simu)

	filetered_data     = analysis.bandpassfilter(np.cos(phase), 32, 48)
	filetered_data     = filetered_data[:,1000*2:]
	phi                = analysis.hilbert_transform(filetered_data)
	PLV                = analysis.plv(phi, num_slice = 10)
	analysis.plot_plv(PLV)



