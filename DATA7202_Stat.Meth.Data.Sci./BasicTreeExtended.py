# BasicTreeExtended.py
import numpy as np
from sklearn.datasets import make_friedman1, make_blobs
from sklearn.model_selection import train_test_split
from enum import Enum
from sklearn.metrics import zero_one_loss

# deferent loss types
class LossType(Enum):
    SQER    = 1
    MISSCLF = 2
    GINI    = 3
    CE      = 4
    
def makedata():
  n_points = 500 # points
 
  X, y =  make_friedman1(n_samples=n_points, n_features=5, 
                         noise=1.0, random_state=100)
         
  return train_test_split(X, y, test_size=0.5, random_state=3)

def makedata_clf():
  n_points = 500 # points
 
  X, y =  make_blobs(n_samples=n_points, centers=2, n_features=5,
                   random_state=10, cluster_std=5)
         
  return train_test_split(X, y, test_size=0.5, random_state=3)
  
# tree node
class TNode:
   def __init__(self, depth, X, y, lossType): 
      global n
      self.depth = depth
      self.X = X   # matrix of explanatory variables
      self.y = y   # vector of response variables
      self.lossType = lossType
      # initialize optimal split parameters
      self.j = None
      self.xi = None
      # initialize children to be None      
      self.left = None
      self.right = None
      # initialize the regional predictor
      self.g = None
      
   def CalculateLoss(self):
       if(len(self.y)==0):
           return 0
       if(self.lossType == LossType.SQER):
           return np.sum(np.power(self.y- self.y.mean(),2))
       
       # count class instances        
       c0 = len(self.y[self.y==0])
       c1 = len(self.y[self.y==1])
       
       if(self.lossType == LossType.MISSCLF):
           return (1 - max(c0,c1)/(c0+c1))*(len(self.y)/n)
       
       p0 = c0/(c0+c1)
       p1 = c1/(c0+c1) 
       if(self.lossType == LossType.GINI):
           return (p0*(1-p0) + p1*(1-p1))*(len(self.y)/n)
       if(self.lossType == LossType.CE):
           if(p0==0 or p1==0):
               return 0
           else:
               return (-0.5*(p0*np.log2(p0) + p1*np.log2(p1)))*(len(self.y)/n)
  
def Construct_Subtree(node, max_depth,lossType):  
    if(node.depth == max_depth or len(node.y) == 1):
        node.g  = node.y.mean()
    else:
        j, xi = CalculateOptimalSplit(node, lossType)               
        node.j = j
        node.xi = xi
        Xt, yt, Xf, yf = DataSplit(node.X, node.y, j, xi)
              
        if(len(yt)>0):
            node.left = TNode(node.depth+1,Xt,yt, lossType)
            Construct_Subtree(node.left, max_depth, lossType)
        
        if(len(yf)>0):        
            node.right = TNode(node.depth+1, Xf,yf, lossType)
            Construct_Subtree(node.right, max_depth, lossType)      
     
    return node

# split the data-set
def DataSplit(X,y,j,xi):
    ids = X[:,j]<=xi      
    Xt  = X[ids == True,:]
    Xf  = X[ids == False,:]
    yt  = y[ids == True]
    yf  = y[ids == False]
    return Xt, yt, Xf, yf             

def CalculateOptimalSplit(node, lossType):
    X = node.X
    y = node.y
    best_var = 0
    best_xi = X[0,best_var]          
    best_split_val = node.CalculateLoss()
    
    m, n  = X.shape
    
    for j in range(0,n):
        for i in range(0,m):
            xi = X[i,j]
            Xt, yt, Xf, yf = DataSplit(X,y,j,xi)
            tmpt = TNode(0, Xt, yt, lossType) 
            tmpf = TNode(0, Xf, yf, lossType) 
            loss_t = tmpt.CalculateLoss()
            loss_f = tmpf.CalculateLoss()    
            curr_val =  loss_t + loss_f
            if (curr_val < best_split_val):
                best_split_val = curr_val
                best_var = j
                best_xi = xi
    return best_var,  best_xi


def Predict(X,node):
    if(node.right == None and node.left != None):
        return Predict(X,node.left)
    
    if(node.right != None and node.left == None):
        return Predict(X,node.right)
    
    if(node.right == None and node.left == None):
        return node.g
    else:
        if(X[node.j] <= node.xi):
            return Predict(X,node.left)
        else:
            return Predict(X,node.right)
  
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = makedata_clf()    
    maxdepth = 10 # maximum tree depth 
    
    n = len(X_train)
    
    # define loss type 
    #lossType = LossType.GINI
    #lossType = LossType.CE
    lossType = LossType.MISSCLF
                
    # Create tree root at depth 0                       
    treeRoot = TNode(0, X_train,y_train,lossType) 
           
    # Build the regression tree with maximal depth equal to max_depth
    Construct_Subtree(treeRoot, maxdepth,lossType) 
        
    # Predict
    y_hat = np.zeros(len(X_test))
    for i in range(len(X_test)):
        y_hat[i] = Predict(X_test[i],treeRoot)          
           
    print("Basic tree: tree loss = ",  zero_one_loss(y_test, np.int64(y_hat)))    