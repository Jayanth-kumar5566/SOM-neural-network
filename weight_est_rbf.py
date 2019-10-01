import som
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Defining train test split for a torch tensor
def split(X,y):
    X=X.numpy()
    y=y.numpy()
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.10)
    X_train=torch.from_numpy(X_train).double()
    X_test=torch.from_numpy(X_test).double()
    y_train=torch.from_numpy(y_train).double()
    y_test=torch.from_numpy(y_test).double()
    return (X_train, X_test, y_train, y_test)

#Defining the training input datasets
X=som.data_train
y=som.label_train

#Converting labels from numpy to torch tesnor
y=torch.from_numpy(y)
y=y.double() #Converting the dtype to float

C=som.C #The weight vector

def Phi(X,sigma,C):
    Phi=[]
    for i in range(X.shape[0]):
        Phi_r=torch.exp(torch.neg(((X[i,:]-C).norm(dim=1)**2)/2*(sigma**2)))
        Phi.append(Phi_r)
    Phi=torch.stack(Phi)
    bias=torch.tensor([1]*X.shape[0],dtype=torch.double).reshape(X.shape[0],1)
    Phi=torch.cat([Phi,bias],dim=1)
    return Phi

#Calculation of the weight
def W(Phi,sigma,y):
    W=torch.mm(torch.mm(torch.inverse(torch.mm(Phi.transpose(0,1),Phi)),Phi.transpose(0,1)),y)
    return W

#Fitting model
def forward(X,y,sigma,C):
    phi=Phi(X,sigma,C)
    w=W(phi,sigma,y)
    return w

#Prediction
def pred(phi,w):
    return torch.mm(phi,w)


#Choosing the best sigma using 10 fold Cross Validation
count=0
opt_sig=[]
opt_loss=[]
sigma=torch.arange(0.001,1,0.001)
while count<10:
    L=[]
    for i in sigma: #Range of search for hyper parameter Sigma
        X_train, X_test, y_train, y_test=split(X,y)
        w=forward(X_train,y_train,i,C)
        loss=(y_test-pred(Phi(X_test,i,C),w)).norm()
        L.append(loss)
        print(i.item(),loss.item())
    min_loss_ind=L.index(min(L))
    opt_loss.append(L[min_loss_ind].item())
    opt_sig.append(sigma[min_loss_ind].item())
    plt.plot(sigma,L,label=str(count)+"- Fold")
    count+=1

plt.xlabel("$\sigma$")
plt.ylabel("Misclassification/Error")
plt.legend()
plt.show()
print "Optimum Sigma is", torch.median(torch.Tensor(opt_sig)).item()


optimum_sigma=0.235

#Classification accuracy of the training data
W=forward(X,y,optimum_sigma,C)
prediction=pred(Phi(X,optimum_sigma,C),W)
prediction_s=torch.sign(prediction) #Converting the prediction values into binary -1 or 1
miscl=(abs(y-prediction_s)/2).sum()
acc=100-((miscl/y.shape[0])*100).item()
print "Classification accuracy of the training data is ",acc

#Predicting labels for the test dataset
x_testing=torch.from_numpy(som.data_test).double()
phi=Phi(x_testing,optimum_sigma,C)
pred_test=pred(phi,W)
pred_test_s=torch.sign(pred_test)
print pred_test_s

file=open("result","w")
for i in pred_test_s:
    file.write(str(i.item())+'\n')

file.close()
