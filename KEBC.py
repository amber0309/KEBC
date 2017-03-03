from __future__ import division
import numpy as np
from scipy.linalg import inv, eig
from scipy.sparse.linalg import eigs
from pylab import *
# from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.stats import gamma
from random import choice

from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import cPickle as pickle

import csv


class KEMDOPERATION:

	@staticmethod
	def kernel_embedding_K(dist, theta, delta):
		Len = len(dist)
		
		m,n = dist[0].shape 
		
		y = np.ones((m,n), dtype = float)
		
		for i in range(0, Len):
			
			d = np.sqrt(dist[i])
			l = delta[0,i]
			a = 0.5
			y1 = np.exp(-dist[i]/(delta[:,i])**2)
		   
			y = y * y1
			
		y = theta * y
		
		return y 
		
	@staticmethod
	def kernel_embedding_D(data, data_sr, feature_type):
	   
		len1  = len(data)
		len2 = len(data_sr)
		
		xx1 = np.transpose(data)
		xx2 = np.transpose(data_sr)
		
		temp = []
		
		for x in xx1:
			temp.append(x.tolist())
		xx1 = temp 
		
		temp = []
		for x in xx2: 
		   temp.append(x.tolist())
		xx2 = temp 
		
		
		num_of_feature = len(feature_type)
		K = []
		#print num_of_feature        
		for i in range(0, num_of_feature):
			K_k = np.zeros((len1, len2), dtype = float)
			K.append(K_k)
		
		dist_x1_x2 = 0.0 
		
		for i in range(0, len1):
			for j in range(0,len2):
				for k in range(0, num_of_feature):
				
					Type = feature_type[k]
					x1 = xx1[k]
					x2 = xx2[k]
				
					if Type == 'numeric':
						dist_x1_x2 = (x1[i] - x2[j]) ** 2 
					elif Type == 'Categorical':
						dist_x1_x2 = float(x1[i]==x2[j])
					else:
						dist_x1_x2 = 0.0 
				
					K[k][i][j] = dist_x1_x2 
		return K 
		
	@staticmethod
	def median_dist(S1, S2, feature_type):
		L1 = len(S1[:,0])
		L2 = len(S2[:,0])
		num_of_feature = len(feature_type)
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			M = []
			for i in range(0, L2):
				for p in range(0, L1):
				
					if feature_type[t] == 'numeric':
						d = np.abs(S1[p,t] - S2[i,t])
					elif feature_type == 'Categorical':
						d = float(S1[p,t] == S2[i,t])
					else: 
						d = 0.0 
				
					M.append(d)
			MM[0,t] = np.median(M)
		return MM
		
	@staticmethod
	def mean_dist(S1, S2, feature_type):
		
		L = len(S1[:,0])
		num_of_feature = len(feature_type)
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			M = []
			for i in range(0, L):
				for p in range(0, i):
				
					if feature_type[t] == 'numeric':
						d = np.abs(S1[p,t] - S2[i,t])
					elif feature_type == 'Categorical':
						d = float(S1[p,t] == S2[i,t])
					else: 
						d = 0.0 
				
					M.append(d)
			MM[0,t] = np.mean(M)
		return MM


def KEBC_cond(X, label_true, score_flg = 0):

	# ----------  initialization  ----------
	N_grp = len(X)
	nclu = len(np.unique(label_true))

	K_x = []
	K_y = []
	Inv_K_x = []
	feature_type_1 = ['numeric']
	feature_type_2 = ['numeric']

	lda = 1e-2
	c = 1

	XY_bck = [i.copy() for i in X]
	# ----------               ----------

	# ---------- Normalization ----------
	for k in range(0, N_grp):
		X[k][:,0] = X[k][:,0] - np.mean(X[k][:,0])
		X[k][:,0] = X[k][:,0] / np.std(X[k][:,0])

		X[k][:,1] = X[k][:,1] - np.mean(X[k][:,1])
		X[k][:,1] = X[k][:,1] / np.std(X[k][:,1])

	# ----------               ----------

	# ---------- kernel matrices within each group ----------
	for keridx in range(0, N_grp):
		xy = X[keridx]
		L, Dim = xy.shape
		x = xy[:,1].reshape(L, 1) #condition on the second column
		y = xy[:,0].reshape(L, 1)

		delta_x = KEMDOPERATION.median_dist(x,x, feature_type_1)
		d_xx = KEMDOPERATION.kernel_embedding_D(x, x, feature_type_1)
		k_xx = KEMDOPERATION.kernel_embedding_K(d_xx, 1, c * delta_x)

		delta_y = KEMDOPERATION.median_dist(y,y, feature_type_2)
		d_yy = KEMDOPERATION.kernel_embedding_D(y, y, feature_type_2)
		k_yy = KEMDOPERATION.kernel_embedding_K(d_yy, 1, c * delta_y)

		I = np.identity(L, dtype = float)
		inv_k_xx = inv(k_xx + lda * L * I)

		K_x.append(k_xx.copy())
		K_y.append(k_yy.copy())
		Inv_K_x.append(inv_k_xx.copy())
	# ----------               ----------

	# ---------- kernel matrices of groups ----------
	D = np.zeros((N_grp, N_grp), dtype = float)
	DD = []
	MD = []

	for i in range(0, N_grp):
		for j in range(0, i):
			xyi = X[i]
			xyj = X[j]

			Li, Dimi = xyi.shape
			xi = xyi[:,1].reshape(Li, 1)
			yi = xyi[:,0].reshape(Li, 1)

			Lj, Dimj = xyj.shape
			xj = xyj[:,1].reshape(Lj, 1)
			yj = xyj[:,0].reshape(Lj, 1)

			d_xij = KEMDOPERATION.kernel_embedding_D(xi, xj, feature_type_1)
			delta_x_ij = KEMDOPERATION.median_dist(xi, xj, feature_type_1)
			k_xij = KEMDOPERATION.kernel_embedding_K(d_xij, 1, c * delta_x_ij)

			d_yij = KEMDOPERATION.kernel_embedding_D(yi, yj, feature_type_2)
			delta_y_ij = KEMDOPERATION.median_dist(yi, yj, feature_type_2)
			k_yij = KEMDOPERATION.kernel_embedding_K(d_yij, 1, c * delta_y_ij)


			term1 = np.trace( np.dot( Inv_K_x[i], np.dot(K_x[i], np.dot(Inv_K_x[i], K_y[i]) ) ) )
			term2 = np.trace( np.dot( Inv_K_x[i], np.dot(k_xij, np.dot(Inv_K_x[j], k_yij.transpose() ) ) )  )
			term3 = np.trace( np.dot( Inv_K_x[j], np.dot(K_x[j], np.dot(Inv_K_x[j], K_y[j]) ) )  )

			if (term1 + term3 - 2 * term2) < 0:
				print 'negtive sqrt'
				return np.zeros((1, N_grp), dtype = float)
			else:
				dist = np.sqrt(term1 + term3 - 2 * term2)

			D[i,j] = dist
			D[j,i] = dist
			MD.append(dist)

	DD.append(D)
	Delta = np.array(np.median(MD)).reshape(1,1)

	K = KEMDOPERATION.kernel_embedding_K(DD, 1, Delta)
	# ----------               ----------

	# ---------- eigen-decomposition ----------
	vals, vecs = eig(K)
	indx = np.argsort(np.real(vals))
	vals = vals[indx]
	vecs = vecs[:,indx]
	# ----------               ----------

	# ---------- conduct k-means++ clustering ----------
	k1 = KMeans(init='k-means++', n_clusters=nclu, n_init=50)
	clu_label = k1.fit_predict( np.real(vecs[:,N_grp-nclu:N_grp]) )
	# ----------                              ----------

	if score_flg == 1:

		# -------------------- beginning of score output --------------------
		true_1 = label_true.count(1)
		true_0 = label_true.count(0)

		ord_1 = np.count_nonzero(clu_label)
		ord_0 = true_1 + true_0 - ord_1

		# ---------- correspondence ----------
		true_clu = membership(label_true, nclu)
		ord_clu = membership(clu_label, nclu)

		ord_cost = np.zeros((nclu,nclu), dtype = float)
		for tr_idx in range(0,nclu):
			for ord_idx in range(0,nclu):
				ord_cost[tr_idx, ord_idx] = 1 - (len( np.intersect1d(true_clu[tr_idx], ord_clu[ord_idx]) )) / len( np.union1d(true_clu[tr_idx], ord_clu[ord_idx]) )
		ord_ridx, ord_cidx = linear_sum_assignment(ord_cost)
		# ---------- correspondence ----------

		if ord_cidx[0] == 0: # 0 in label_true corresponds to 0 in clu_label
			# so 1 in clu_label means edge exists
			samelabel = 1
			z0 = (np.matrix(clu_label)).T # cluster with edge, group = 1 in this cluster 
			z1 = 1 - z0 # cluster without edge

			l0 = 1 / ord_1
			l1 = 1 / ord_0

		else: # cluster_0 in label_true corresponds to cluster_1 in clu_label
			# so 0 in clu_label means edge exists
			samelabel = 0
			z1 = (np.matrix(clu_label)).T # cluster without edge
			z0 = 1 - z1 # cluster with edge

			l0 = 1 / ord_0
			l1 = 1 / ord_1

		Z_H1 = np.hstack((z0, z0))
		Z_H2 = np.hstack((z1, z1))

		L_H1 = np.zeros((2,2))
		L_H1[0,0] = l0
		L_H1[1,1] = l0

		L_H2 = np.zeros((2,2))
		L_H2[0,0] = l1
		L_H2[1,1] = l1

		# ----- k_h1 -----
		Y_H1 = np.ones((nclu, N_grp)) / nclu
		ZLZT_H1 = np.dot(np.dot(Z_H1, L_H1), Y_H1)
		K_H1 = K - np.dot(K, ZLZT_H1) - np.dot(ZLZT_H1.T, K) + np.dot(np.dot(ZLZT_H1.T, K), ZLZT_H1)
		# -----      -----

		# ----- k_h2 -----
		Y_H2 = np.ones((nclu, N_grp)) / nclu
		ZLZT_H2 = np.dot(np.dot(Z_H2, L_H2), Y_H2)
		K_H2 = K - np.dot(K, ZLZT_H2) - np.dot(ZLZT_H2.T, K) + np.dot(np.dot(ZLZT_H2.T, K), ZLZT_H2)
		# -----      -----

		outfile = open('score.txt', 'wb')
		for i in range(0, N_grp):
			score = K_H1[i,i] / K_H2[i,i]
			outfile.write(str(score) + ' '+str(label_true[i]) + '\n')
		# -------------------- end of score output --------------------

	return clu_label

def KEBC_cond_ref(XY, label_true, score_flg = 0):
	# ----------  initialization  ----------
	N_grp = len(XY)
	nclu = len(np.unique(label_true))

	K_x = []
	K_y = []
	Inv_K_x = []
	Delta_x = []
	Delta_y = []
	Sigma = []
	LL = []
	feature_type_1 = ['numeric']
	feature_type_2 = ['numeric']
	lda = 1e-2

	XY_bck = [i.copy() for i in XY]
	# ----------               ----------

	# ---------- Normalization ----------
	for k in range(0, N_grp):
		XY[k][:,0] = XY[k][:,0] - np.mean(XY[k][:,0])
		XY[k][:,0] = XY[k][:,0] / np.std(XY[k][:,0])

		XY[k][:,1] = XY[k][:,1] - np.mean(XY[k][:,1])
		XY[k][:,1] = XY[k][:,1] / np.std(XY[k][:,1])
	# ----------               ----------

	# ---------- kernel matrices within each group ----------
	for keridx in range(0, N_grp):
		xy = XY[keridx]
		L, Dim = xy.shape
		x = xy[:,0].reshape(L, 1)
		y = xy[:,1].reshape(L, 1)

		delta_x = KEMDOPERATION.median_dist(x,x, feature_type_1)
		delta_y = KEMDOPERATION.median_dist(y,y, feature_type_2)

		d_xx = KEMDOPERATION.kernel_embedding_D(x, x, feature_type_1)
		k_xx = KEMDOPERATION.kernel_embedding_K(d_xx, 1, delta_x)

		d_yy = KEMDOPERATION.kernel_embedding_D(y, y, feature_type_2)
		k_yy = KEMDOPERATION.kernel_embedding_K(d_yy, 1, delta_y)

		K_x.append(k_xx)
		K_y.append(k_yy)

		LL.append(L)

		Delta_x.append(delta_x)
		Delta_y.append(delta_y)
	# ----------  ----------

	# ---------- kernel matrices with reference group ----------
	xy0 = XY[0]
	L0, Dim0 = xy0.shape
	x0 = xy0[:,0].reshape(L0, 1)
	y0 = xy0[:,1].reshape(L0, 1)
	for bu in range(0, N_grp):
		xybu = XY[bu]

		Lbu, Dimbu = xybu.shape
		xbu = xybu[:,0].reshape(Lbu, 1)
		ybu = xybu[:,1].reshape(Lbu, 1)
		I = np.identity(Lbu, dtype = float)

		d_xbu = KEMDOPERATION.kernel_embedding_D(xbu, x0, feature_type_1)
		delta_xbu = KEMDOPERATION.median_dist(xbu, x0, feature_type_1)
		k_xbu = KEMDOPERATION.kernel_embedding_K(d_xbu, 1, delta_xbu)

		YI = np.ones((L0, 1), dtype = float)
		alpha_bu = np.dot( np.dot( inv(K_x[bu] + lda * I), k_xbu), YI ) * Lbu / L0

		sig = np.zeros((Lbu, Lbu), dtype = float)
		for tem in range(0, Lbu):
			sig[tem, tem] = np.sqrt(max(alpha_bu[tem], 0))

		Sigma.append(sig)
	# ----------   ----------

	# ---------- inverse operator ----------
	for invidx in range(0, N_grp):

		I = np.identity(LL[invidx], dtype = float)
		k_xx = K_x[invidx]
		inv_k_xx = inv( np.dot(np.dot(Sigma[invidx], k_xx), Sigma[invidx]) + lda * LL[invidx] * I )

		Inv_K_x.append(inv_k_xx)
	# ----------   ----------

	# ---------- kernel matrices of groups ----------
	D = np.zeros((N_grp, N_grp), dtype = float)
	DD = []
	MD = []

	for i in range(0, N_grp):
		for j in range(0, i):
			xyi = XY[i]
			xyj = XY[j]

			Li, Dimi = xyi.shape
			xi = xyi[:,0].reshape(Li, 1)
			yi = xyi[:,1].reshape(Li, 1)

			Lj, Dimj = xyj.shape
			xj = xyj[:,0].reshape(Lj, 1)
			yj = xyj[:,1].reshape(Lj, 1)

			d_xij = KEMDOPERATION.kernel_embedding_D(xi, xj, feature_type_1)
			delta_x_ij = KEMDOPERATION.median_dist(xi, xj, feature_type_1)
			k_xij = KEMDOPERATION.kernel_embedding_K(d_xij, 1, delta_x_ij)

			d_yij = KEMDOPERATION.kernel_embedding_D(yi, yj, feature_type_2)
			delta_y_ij = KEMDOPERATION.median_dist(yi, yj, feature_type_2)
			k_yij = KEMDOPERATION.kernel_embedding_K(d_yij, 1, delta_y_ij)

			term1 = np.trace( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot(Sigma[i], Inv_K_x[i]), Sigma[i]), K_y[i]), Sigma[i]), Inv_K_x[i]), Sigma[i]), K_x[i]) )
			term2 = np.trace( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot(Sigma[i], Inv_K_x[i]), Sigma[i]), k_yij), Sigma[j]), Inv_K_x[j]), Sigma[j]), k_xij.transpose()) )
			term3 = np.trace( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot( np.dot(Sigma[j], Inv_K_x[j]), Sigma[j]), K_y[j]), Sigma[j]), Inv_K_x[j]), Sigma[j]), K_x[j]) )

			if (term1 + term3 - 2 * term2) < 0:
				print 'negtive sqrt'
				return np.zeros((1, N_grp), dtype = float)
			else:
				dist = np.sqrt(term1 + term3 - 2 * term2)

			#dist = np.sqrt(term1 + term3 - 2 * term2)
			D[i,j] = dist
			D[j,i] = dist
			MD.append(dist)

	DD.append(D)
	Delta = np.array(np.median(MD)).reshape(1,1)#/5.0

	K = KEMDOPERATION.kernel_embedding_K(DD, 1, Delta)
	# ----------   ----------

	# ---------- eigen-decomposition ----------
	vals, vecs = eig(K)
	indx = np.argsort(np.real(vals))
	vals = vals[indx]
	vecs = vecs[:,indx]
	# ----------   ----------

	# ---------- k-means++ clustering ----------
	k2 = KMeans(init='k-means++', n_clusters=nclu, n_init=50)
	clu_label = k2.fit_predict( np.real(vecs[:,N_grp-nclu:N_grp]) )
	# ----------   ----------

	if score_flg == 1:
		# -------------------- beginning of score output --------------------
		true_1 = label_true.count(1)
		true_0 = label_true.count(0)

		ord_1 = np.count_nonzero(clu_label)
		ord_0 = true_1 + true_0 - ord_1

		# ---------- correspondence ----------
		true_clu = membership(label_true, nclu)
		ord_clu = membership(clu_label, nclu)

		ord_cost = np.zeros((nclu,nclu), dtype = float)
		for tr_idx in range(0,nclu):
			for ord_idx in range(0,nclu):
				ord_cost[tr_idx, ord_idx] = 1 - (len( np.intersect1d(true_clu[tr_idx], ord_clu[ord_idx]) )) / len( np.union1d(true_clu[tr_idx], ord_clu[ord_idx]) )
		ord_ridx, ord_cidx = linear_sum_assignment(ord_cost)
		# ---------- correspondence ----------

		if ord_cidx[0] == 0: # 0 in label_true corresponds to 0 in clu_label
			# so 1 in clu_label means edge exists
			samelabel = 1
			z0 = (np.matrix(clu_label)).T # cluster with edge, group = 1 in this cluster 
			z1 = 1 - z0 # cluster without edge

			l0 = 1 / ord_1
			l1 = 1 / ord_0

		else: # cluster_0 in label_true corresponds to cluster_1 in clu_label
			# so 0 in clu_label means edge exists
			samelabel = 0
			z1 = (np.matrix(clu_label)).T # cluster without edge
			z0 = 1 - z1 # cluster with edge

			l0 = 1 / ord_0
			l1 = 1 / ord_1

		Z_H1 = np.hstack((z0, z0))
		Z_H2 = np.hstack((z1, z1))

		L_H1 = np.zeros((2,2))
		L_H1[0,0] = l0
		L_H1[1,1] = l0

		L_H2 = np.zeros((2,2))
		L_H2[0,0] = l1
		L_H2[1,1] = l1

		# ----- k_h1 -----
		Y_H1 = np.ones((nclu, N_grp)) / nclu
		ZLZT_H1 = np.dot(np.dot(Z_H1, L_H1), Y_H1)
		K_H1 = K - np.dot(K, ZLZT_H1) - np.dot(ZLZT_H1.T, K) + np.dot(np.dot(ZLZT_H1.T, K), ZLZT_H1)
		# -----      -----

		# ----- k_h2 -----
		Y_H2 = np.ones((nclu, N_grp)) / nclu
		ZLZT_H2 = np.dot(np.dot(Z_H2, L_H2), Y_H2)
		K_H2 = K - np.dot(K, ZLZT_H2) - np.dot(ZLZT_H2.T, K) + np.dot(np.dot(ZLZT_H2.T, K), ZLZT_H2)
		# -----      -----

		outfile = open('score.txt', 'wb')
		for i in range(0, N_grp):
			score = K_H1[i,i] / K_H2[i,i]
			outfile.write(str(score) + ' '+str(label_true[i]) + '\n')
		# -------------------- end of score output --------------------

	return clu_label

def KEBC_marg(X, label_true, score_flg = 0):

	# ----------  initialization  ----------
	N_grp = len(X)
	nclu = len(np.unique(label_true))

	D_x = []
	Delta_x = []
	feature_type_1 = ['numeric']

	lda = 1e-2

	XY_bck = [i.copy() for i in X]
	# ----------               ----------

	# ---------- Normalization ----------
	for k in range(0, N_grp):
		X[k][:,0] = X[k][:,0] - np.mean(X[k][:,0])
		X[k][:,0] = X[k][:,0] / np.std(X[k][:,0])
	# ----------               ----------

	# ---------- distance matrices within each group ----------
	for keridx in range(0, N_grp):
		x = X[keridx]
		L, Dim = x.shape
		x = x[:,0].reshape(L, 1)

		delta_x = KEMDOPERATION.median_dist(x,x, feature_type_1)

		d_xx = KEMDOPERATION.kernel_embedding_D(x, x, feature_type_1)
		# k_xx = KEMDOPERATION.kernel_embedding_K(d_xx, 1, delta_x)

		D_x.append(d_xx)

		Delta_x.append(delta_x)
	# ----------               ----------

	# ---------- kernel width ----------
	delta_x = np.median(Delta_x)

	K_x = []

	sigmax = Delta_x[0]
	sigmax[0][0] = np.mean(Delta_x)
	# ----------               ----------

	# ---------- kernel matrices within each group ----------
	for k in range(0, N_grp):
		k_xx = KEMDOPERATION.kernel_embedding_K(D_x[k], 1, sigmax)
		K_x.append(k_xx.copy())
	# ----------               ----------

	# ---------- kernel matrices of groups ----------
	D = np.zeros((N_grp, N_grp), dtype = float)
	DD = []
	MD = []

	for i in range(0, N_grp):
		for j in range(0, i):
			xi = X[i]
			xj = X[j]

			Li, Dimi = xi.shape

			Lj, Dimj = xj.shape

			d_xij = KEMDOPERATION.kernel_embedding_D(xi, xj, feature_type_1)
			k_xij = KEMDOPERATION.kernel_embedding_K(d_xij, 1, sigmax)

			term1 = np.sum(np.sum(K_x[i])) / Li / Li
			term2 = np.sum(np.sum(k_xij)) / Li / Lj
			term3 = np.sum(np.sum(K_x[j])) / Lj / Lj

			if (term1 + term3 - 2 * term2) < 0:
				print 'negtive sqrt'
				return np.zeros((1, N_grp), dtype = float)
			else:
				dist = np.sqrt(term1 + term3 - 2 * term2)

			D[i,j] = dist
			D[j,i] = dist
			MD.append(dist)

	DD.append(D)
	Delta = np.array(np.mean(MD)).reshape(1,1) * 2.0#/5.0

	K = KEMDOPERATION.kernel_embedding_K(DD, 1, Delta)
	# ----------               ----------

	# ---------- eigen-decomposition ----------
	vals, vecs = eig(K)
	indx = np.argsort(np.real(vals))
	vals = vals[indx]
	vecs = vecs[:,indx]
	# ----------               ----------

	# ---------- k-means++ clustering ----------
	k1 = KMeans(init='k-means++', n_clusters=nclu, n_init=50)
	clu_label = k1.fit_predict( np.real(vecs[:,N_grp-nclu:N_grp]) )

	if score_flg == 1:
		# -------------------- beginning of score output --------------------
		true_1 = label_true.count(1)
		true_0 = label_true.count(0)

		ord_1 = np.count_nonzero(clu_label)
		ord_0 = true_1 + true_0 - ord_1

		# ---------- correspondence ----------
		true_clu = membership(label_true, nclu)
		ord_clu = membership(clu_label, nclu)

		ord_cost = np.zeros((nclu,nclu), dtype = float)
		for tr_idx in range(0,nclu):
			for ord_idx in range(0,nclu):
				ord_cost[tr_idx, ord_idx] = 1 - (len( np.intersect1d(true_clu[tr_idx], ord_clu[ord_idx]) )) / len( np.union1d(true_clu[tr_idx], ord_clu[ord_idx]) )
		ord_ridx, ord_cidx = linear_sum_assignment(ord_cost)
		# ---------- correspondence ----------

		if ord_cidx[0] == 0: # 0 in label_true corresponds to 0 in clu_label
			# so 1 in clu_label means edge exists
			samelabel = 1
			z0 = (np.matrix(clu_label)).T # cluster with edge, group = 1 in this cluster 
			z1 = 1 - z0 # cluster without edge

			l0 = 1 / ord_1
			l1 = 1 / ord_0

		else: # cluster_0 in label_true corresponds to cluster_1 in clu_label
			# so 0 in clu_label means edge exists
			samelabel = 0
			z1 = (np.matrix(clu_label)).T # cluster without edge
			z0 = 1 - z1 # cluster with edge

			l0 = 1 / ord_0
			l1 = 1 / ord_1

		Z_H1 = np.hstack((z0, z0))
		Z_H2 = np.hstack((z1, z1))

		L_H1 = np.zeros((2,2))
		L_H1[0,0] = l0
		L_H1[1,1] = l0

		L_H2 = np.zeros((2,2))
		L_H2[0,0] = l1
		L_H2[1,1] = l1

		# ----- k_h1 -----
		Y_H1 = np.ones((nclu, N_grp)) / nclu
		ZLZT_H1 = np.dot(np.dot(Z_H1, L_H1), Y_H1)
		K_H1 = K - np.dot(K, ZLZT_H1) - np.dot(ZLZT_H1.T, K) + np.dot(np.dot(ZLZT_H1.T, K), ZLZT_H1)
		# -----      -----

		# ----- k_h2 -----
		Y_H2 = np.ones((nclu, N_grp)) / nclu
		ZLZT_H2 = np.dot(np.dot(Z_H2, L_H2), Y_H2)
		K_H2 = K - np.dot(K, ZLZT_H2) - np.dot(ZLZT_H2.T, K) + np.dot(np.dot(ZLZT_H2.T, K), ZLZT_H2)
		# -----      -----

		outfile = open('score.txt', 'wb')
		for i in range(0, N_grp):
			score = K_H1[i,i] / K_H2[i,i]
			outfile.write(str(score) + ' '+str(label_true[i]) + '\n')
		# -------------------- end of score output --------------------

	return clu_label

def membership(label, number_cluster):
	clu_dict = {}
	for i in range(0, number_cluster):
		MB = []
		for k in range(0, len(label)):
			if label[k] == i:
				MB.append(k)
		print "Cluster" + str(i) + ":", MB
		print "\n"
		clu_dict[i] = MB
	return clu_dict


if __name__ == '__main__':
	Max = 1
	count = 0
	nclu = 2
	Ngrp = 100

	re_all = np.zeros((1,2 * nclu), dtype = float)
	re_sum = np.zeros((1,2 * nclu), dtype = float)
	for i in range(0,Max):
		# re = exp5(2, 20)
		re = exp_synth(nclu, Ngrp)
		print re
		if (re == 0).all():
			pass
		else:
			re_sum = re_sum + re
			re_all = np.vstack((re_all, re))
			count += 1
			print 'i = %d' %i
	re_sum = re_sum / count
	print 'mean vector'
	print re_sum
	print '\n'
	re_all = np.delete(re_all, (0), axis = 0)
	print 'std vector'
	print np.std(re_all, axis = 0)
	print '\n'
	print 'count = %d' %count