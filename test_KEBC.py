"""
A toy synthetic test for KEBC
(Anaconda 4.3.0 64-bit for Windows)

Shoubo (shoubo.hu AT gmail.com)
06/02/2017

USAGE:
  exp_synth(n_clu, n_grp)

 
INPUT:
  n_clu      - number of clusters
  n_grp      - number of groups
  
"""
from __future__ import division
import numpy as np
from scipy.linalg import inv, eig
from scipy.optimize import linear_sum_assignment
from scipy.stats import gamma
from random import choice
from KEBC_bochner import KEBC_cond_bochner, membership

from sklearn.cluster import KMeans

def gen_stat(lbl_true, lbl_ord, n_clu):
	label_true = lbl_true
	label_ord = lbl_ord

	num_ord = np.unique(label_ord)

	if (len(num_ord) < n_clu):
		return np.zeros((1, 4 * n_clu), dtype = float)

	true_clu = membership(label_true, n_clu)
	ord_clu = membership(label_ord, n_clu)

	ord_cost = np.zeros((n_clu,n_clu), dtype = float)
	for tr_idx in range(0,n_clu):
		for ord_idx in range(0,n_clu):
			ord_cost[tr_idx, ord_idx] = 1 - (len( np.intersect1d(true_clu[tr_idx], ord_clu[ord_idx]) )) / len( np.union1d(true_clu[tr_idx], ord_clu[ord_idx]) )
	ord_ridx, ord_cidx = linear_sum_assignment(ord_cost)

	prcs_ord = []
	recl_ord = []
	for pre_idx in range(0,n_clu):
		prcs_tem_ord = ( len(np.intersect1d(true_clu[ord_ridx[pre_idx]], ord_clu[ord_cidx[pre_idx]])) ) / (len( ord_clu[ord_cidx[pre_idx]] ))
		prcs_ord.append(prcs_tem_ord)

		recl_tem_ord = ( len(np.intersect1d(true_clu[ord_ridx[pre_idx]], ord_clu[ord_cidx[pre_idx]])) ) / (len( true_clu[ord_ridx[pre_idx]] ))
		recl_ord.append(recl_tem_ord)

	result = np.hstack((prcs_ord, recl_ord))

	return result


def Generate_XY(label, sample_size):
	ncoeff = 0.1
	
	Px = 20 * np.random.rand() - 10
	Py = 20 * np.random.rand() - 10

	wt = np.random.rand(3) + 0.5
	wt = wt/np.sum(wt)

	L1 = int(wt[0] * sample_size)
	x1 = 0.3 * np.random.randn(L1, 1) - 1
	L2 = int(wt[1] * sample_size)
	x2 = 0.3 * np.random.randn(L2, 1) + 1
	L3 = sample_size - L1 - L2
	x3 = 0.3 * np.random.randn(L3, 1)

	x = np.concatenate((x1, x2, x3), axis = 0)
	c = 0.4 * np.random.rand(1) + 0.4

	if label == 0:
		n = np.random.randn(sample_size, 1)
		y = np.exp(c * x) + n * ncoeff
	elif label == 1:
		n = - np.random.rand(sample_size, 1)
		y = np.cos(c * x * n) + n * ncoeff
	elif label == 2:
		n = - np.random.rand(sample_size, 1)
		y = np.cos(c * x * n) * n * ncoeff
		n = np.random.rand(sample_size ,1)
		y = np.sign(c * x) * ((c * x)**2) + n * ncoeff

	else:
		pass

	x = x + Px
	y = y + Py

	xy = np.concatenate((x, y), axis = 1)

	return xy

def exp_synth(n_clu, n_grp):
	N = n_grp - n_clu
	XY = []
	label_true = []

	for init in range(0, n_clu):
		label_true.append(init)
		sample_size = choice(np.arange(40,50))
		xy = Generate_XY(init, sample_size)
		XY.append(xy)

	for i in range(0, N):
		label = choice(np.arange(0, n_clu))
		label_true.append(label)
		sample_size = choice(np.arange(40, 50))
		xy = Generate_XY(label, sample_size)

		XY.append(xy)

	label = KEBC_cond_bochner([i.copy() for i in XY], label_true, 100, 0)

	result = gen_stat(label_true, label, n_clu)

	print '[precisoin, recall]'
	print result

	return result


if __name__ == '__main__':
	exp_synth(2,50)
