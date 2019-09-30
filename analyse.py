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
#Find the center vectors using SOM
xi=data_train[0,:] #need to add randomnesss
#Creating a random weight vector of size (5x4x33)
W=torch.rand(5,4,33,dtype=torch.float64)
#Compute discriminant function on the neurons uses euclidean norm
D=(W-xi).norm(dim=2)
#Selecting the Best matching unit
bmu=(D==D.min()).nonzero() #Python indexes start from 0, this indicates the location
#Function that creates a Grid
from itertools import product
coordinates = numpy.array(list(product(xrange(5), xrange(4))))
coordinates=torch.from_numpy(coordinates)
#Functtion that creates euclidean distance matrix on the grid
dist_mat=((coordinates-bmu)**2).double().norm(dim=1).reshape(5,4)
#Function that creates the neighbourhood function
sigma=1
neigh=torch.exp(torch.neg(torch.div(dist_mat**2, 2*(sigma**2))))
#Weight adaptation
eta=0.001
W=W+eta*neigh.reshape([5,4,1])*(xi-W)
