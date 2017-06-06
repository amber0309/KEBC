"""
KEBC (Kernel embedding-based clustering) python implementation
(Anaconda 4.3.0 64-bit for python 2.7 for Windows)

Shoubo (shoubo.hu AT gmail.com)
06/06/2017

USAGE:
  label = KEBC_cond_bochner(XY, label_true, k, score_flg)
  label = KEBC_marg_bochner(XY, label_true, k, score_flg)
 
INPUT:
  XY         - input data, list of numpy arrays. rows of each array are 
               i.i.d. samples, column of each array represent variables
  label_true - the ground truth of cluster label of each group
  k          - length of explicit kernel mapping
  score_flg  - output score file or not. 1 - yes, 0 - no
 
OUTPUT: 
  label      - list of cluster label for each group
 
"""
from __future__ import division
import numpy as np
from scipy.linalg import inv, eig
from scipy.sparse.linalg import eigs
from scipy.optimize import linear_sum_assignment
from scipy.stats import gamma
from random import choice
from sklearn.cluster import KMeans


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

def KEBC_marg_bochner(X, label_true, k, score_flg = 0):
	
	N_grp = len(X)
	nclu = len(np.unique(label_true))

	feature_type_1 = ['numeric']
	feature_type_2 = ['numeric']

	mu_list = []

	for i in range(0, N_grp):
		x = X[i]
		L, Dim = x.shape
		x = x[:,0].reshape(1, L)

		delta_x = KEMDOPERATION.median_dist(x.T,x.T,feature_type_1)

		omega_x = delta_x[0,0] * np.random.randn(k,1)

		phi_x_temp = omega_x.dot(x)
		phi_x = np.zeros((2*k,L), dtype = float)
		phi_x[::2,:] = np.cos(phi_x_temp)
		phi_x[1::2,:] = np.sin(phi_x_temp)
		phi_x = 1/np.sqrt(k) * phi_x
		mu_x = np.mean(phi_x, axis = 1).reshape(-1,1)

		mu_list.append(mu_x)

	D = np.zeros((N_grp, N_grp), dtype = float)
	DD = []
	MD = []

	for i in range(0, N_grp):
		for j in range(0, i):
			dist = np.trace( (mu_list[i] - mu_list[j]).T.dot(mu_list[i] - mu_list[j]) )

			D[i,j] = dist
			D[j,i] = dist
			MD.append(dist)

	DD.append(D)
	Delta = np.array(np.median(MD)).reshape(1,1)

	K = KEMDOPERATION.kernel_embedding_K(DD, 1, Delta)
	vals, vecs = eigsh(K, k=nclu, which='LM')

	k1 = KMeans(init='k-means++', n_clusters=nclu, n_init=50)
	clu_label = k1.fit_predict( np.real(vecs) )

	# -------------------- beginning of score output --------------------
	if score_flg == 1:
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

		outfile = open('score_marg_k' + str(k) + '.txt', 'wb')
		for i in range(0, N_grp):
			score = K_H1[i,i] / K_H2[i,i]
			outfile.write(str(score) + ' '+str(label_true[i]) + '\n')
	# -------------------- end of score output --------------------

	return clu_label


def KEBC_cond_bochner(X, label_true, k, score_flg = 0):
	
	N_grp = len(X)
	nclu = len(np.unique(label_true))

	U = []
	feature_type_1 = ['numeric']
	feature_type_2 = ['numeric']

	lda = 1e-2

	for i in range(0, N_grp):
		
		xy = X[i]

		L, Dim = xy.shape 

		I = np.identity(2*k, dtype = float)

		x = xy[:,1].reshape(1,L)
		y = xy[:,0].reshape(1,L)

		delta_x = KEMDOPERATION.median_dist(x.T,x.T,feature_type_1)
		delta_y = KEMDOPERATION.median_dist(y.T,y.T,feature_type_2)

		omega_x = delta_x[0,0] * np.random.randn(k,1)
		omega_y = delta_y[0,0] * np.random.randn(k,1)

		phi_x_temp = omega_x.dot(x)
		phi_x = np.zeros((2*k,L), dtype = float)
		phi_x[::2,:] = np.cos(phi_x_temp)
		phi_x[1::2,:] = np.sin(phi_x_temp)
		phi_x = 1/np.sqrt(k) * phi_x 


		phi_y_temp = omega_y.dot(y)
		phi_y = np.zeros((2*k,L), dtype = float)
		phi_y[::2,:] = np.cos(phi_y_temp)
		phi_y[1::2,:] = np.sin(phi_y_temp)
		phi_y = 1/np.sqrt(k) * phi_y 


		P_i = phi_y.dot(phi_x.T)/L
		Pi_i = inv(phi_x.dot(phi_x.T)/L + lda * I)

		U_i = P_i.dot(Pi_i) 
		U.append(U_i)

	D = np.zeros((N_grp, N_grp), dtype = float)
	DD = []
	MD = []

	for i in range(0, N_grp):
		for j in range(0, i):

			dist = np.sqrt( np.trace( (U[i] - U[j]).T.dot(U[i] - U[j])) )

			D[i,j] = dist
			D[j,i] = dist
			MD.append(dist)

	DD.append(D)
	Delta = np.array(np.mean(MD)).reshape(1,1)

	K = KEMDOPERATION.kernel_embedding_K(DD, 1, Delta)
	vals, vecs = eigsh(K, k=nclu, which='LM')

	k1 = KMeans(init='k-means++', n_clusters=nclu, n_init=50)
	clu_label = k1.fit_predict( np.real(vecs) )

	# -------------------- beginning of score output --------------------
	if score_flg == 1:

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

		outfile = open('score_cond_k' + str(k) + '.txt', 'wb')
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
