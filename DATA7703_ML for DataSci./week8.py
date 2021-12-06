#(c)
import numpy as np
import torch
torch.manual_seed(1)
x = torch.rand(3,3)

print(x.t()@x + torch.ones_like(x))
print(x.int())
print(torch.relu(x))


#%%(d)
import matplotlib.pyplot as plt
x = torch.randn(100,2, requires_grad=True)
plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy())
plt.show()


#%%(e) write down the derivative of sigmoid function
x = torch.linspace(-5,5,1000)
y = torch.sigmoid(x)
plt.plot(x, y*(1-y))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#%%(f)
x = torch.linspace(-5,5,1000, requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()

plt.plot(x.detach().numpy(), x.grad)
plt.xlabel('x')
plt.ylabel('x_grad')
plt.show()

#%%(g)
import torch
import torch.nn as nn
torch.manual_seed(1)
x = torch.randn(3,4)
net = nn.Linear(4,1)
print(net(x))

list(net.parameters())

#%%(Q2)
import torch

def regression_data(n=500, d=2):
    X = torch.rand(n, d)
    w = torch.rand(d+1)
    Y = X@w[1:]+w[0]+torch.rand(n)*0.1
    return X, Y

torch.manual_seed(1)
X, Y = regression_data(n=200, d=2)

#scatter plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

#view from another angle
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
ax.view_init(30,10)
plt.show()

#%%(b)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, Y)
print(reg.intercept_, reg.coef_)

#%%(c)
import torch.optim as optim
from torch.autograd import Variable

def OLS(X,Y):
    X = torch.cat([torch.ones(X.shape[0],1),X], dim=1)
    w = torch.zeros(X.shape[1], requires_grad=True)
    optimizer = optim.SGD([w], lr=0.5, momentum=0)
    for i in range(200):
        optimizer.zero_grad()
        loss = torch.mean((X @ w - Y)**2)
        loss.backward()
        optimizer.step()
    print(w)

OLS(X,Y)

#%%
import torch.optim as optim
import torch.nn as nn
from torch.nn.modules.loss import  MSELoss

def OLS2(X, Y, niter=200, lr=0.5):
    Y = Y.reshape(-1,1)
    net = nn.Linear(2,1)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0)
    mse = MSELoss()
    for i in range(niter):
        optimizer.zero_grad()
        loss = mse(net(X), Y)
        loss.backward()
        optimizer.step()
    for param in net.parameters():
        print(param)

OLS2(X,Y)

#%%(d)

OLS2(X, Y, lr=0.01)
OLS2(X, Y, lr=0.1)
OLS2(X, Y, lr=1)