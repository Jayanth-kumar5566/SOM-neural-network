#Importing necessary modules
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

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
sigma=1
eta=0.001
sam_dim=data_train.shape[1] #Sample dimension

#Creating a random weight vector of size (5x4x33)
W=torch.rand(lattice_size[0],lattice_size[1],sam_dim,dtype=torch.float64)

#Creating a Grid
from itertools import product
coordinates = numpy.array(list(product(xrange(lattice_size[0]), xrange(lattice_size[1]))))
coordinates=torch.from_numpy(coordinates)

#Selecting training Samples
ind=numpy.random.randint(0,data_train.shape[0]+1) #Chooses random index
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
