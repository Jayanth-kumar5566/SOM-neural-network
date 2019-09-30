#Importing necessary modules
from __future__ import division
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt

#Loading the dataset
data_train=scipy.io.loadmat("data_train.mat")
data_train=data_train["data_train"]
data_test=scipy.io.loadmat("data_test.mat")
data_test=data_test["data_test"]
label_train=scipy.io.loadmat("label_train.mat")
label_train=label_train["label_train"]

#Converting the training data from numpy to torch tensor
data_train=torch.from_numpy(data_train)

#Defining Hyper parameters
lattice_size=[5,4]
sigma_i=5 #the diagonal distance of the lattice
eta_i=0.1
sam_dim=data_train.shape[1] #Sample dimension
max_iter_tot=1500
max_iter_sop=1000
t1=max_iter_sop/numpy.log(sigma_i)
t2=max_iter_sop

#defining functions of sigma and eta
def sigma_iter(iter,t1,sigma):
	return sigma*numpy.exp(-iter/t1)

def eta_iter(iter,t2,eta):
	return eta*numpy.exp(-iter/t1)

#Creating a random weight vector of size (5x4x33)
W=torch.rand(lattice_size[0],lattice_size[1],sam_dim,dtype=torch.float64)

#Creating a Grid
from itertools import product
coordinates = numpy.array(list(product(xrange(lattice_size[0]), xrange(lattice_size[1]))))
coordinates=torch.from_numpy(coordinates)

#Metrics
W_pre = W.clone().detach()
c_W=[] #Change in Weights

#Looping
count=0
while count<max_iter_tot:
	#The two phases
	if count<max_iter_sop:
		print "Self Organizing Phase"
		sigma=sigma_iter(count,t1,sigma_i)
		eta=eta_iter(count,t2,eta_i)
	else:
		print "Convergence Phase"
		sigma=0.01
		eta=0.01
	#Selecting training Samples
	ind=numpy.random.randint(0,data_train.shape[0]) #Chooses random index
	xi=data_train[ind,:] 

	#Compute discriminant function on the neurons uses euclidean norm
	D=(W-xi).norm(dim=2)

	#Selecting the Best matching unit
	bmu=(D==D.min()).nonzero() #Python indexes start from 0, this indicates the location

	#Creating euclidean distance matrix on the grid
	dist_mat=((coordinates-bmu)**2).double().norm(dim=1).reshape(lattice_size[0],lattice_size[1])

	#Creating the neighbourhood function
	neigh=torch.exp(torch.neg(torch.div(dist_mat**2, 2*(sigma**2))))

	#Weight adaptation
	W=W+eta*neigh.reshape([lattice_size[0],lattice_size[1],1])*(xi-W)
	
	c_W.append(((W-W_pre)**2).norm().item())
	W_pre = W.clone().detach()
	count+=1

#plot of the metric
plt.plot(range(max_iter_tot),c_W)
plt.xlabel("Number of iterations")
plt.ylabel("Change in the weights")
plt.show()

#The Center vectors are 
C=W.reshape([20,33])
