#%% Q1
# BasicTreeExtended.py
import numpy as np
from sklearn.datasets import make_friedman1, make_blobs
from sklearn.model_selection import train_test_split
from enum import Enum
from sklearn.metrics import zero_one_loss

# different loss types
class LossType(Enum):
    SQER = 1
    MISSCLF = 2
    GINI = 3
    CE = 4

def makedata():
    n_points = 500  # points

    X, y = make_friedman1(n_samples=n_points, n_features=5,
                          noise=1.0, random_state=100)

    return train_test_split(X, y, test_size=0.5, random_state=3)

def makedata_clf():
    n_points = 500  # points

    X, y = make_blobs(n_samples=n_points, centers=2, n_features=5,
                      random_state=10, cluster_std=5)

    return train_test_split(X, y, test_size=0.5, random_state=3)

# tree node
class TNode:
    def __init__(self, depth, X, y, lossType):
        global n
        self.depth = depth
        self.X = X  # matrix of explanatory variables
        self.y = y  # vector of response variables
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
        if (len(self.y) == 0):
            return 0

        #Entropy impurity
        if (self.lossType == LossType.SQER):
            return np.sum(np.power(self.y - self.y.mean(), 2))

        # count class instances
        c0 = len(self.y[self.y == 0]) #count of zeroz
        c1 = len(self.y[self.y == 1]) #count of ones

        if (self.lossType == LossType.MISSCLF):
            return (1 - max(c0, c1) / (c0 + c1)) * (len(self.y) / n) #scaled by whole dataset


        p0 = c0 / (c0 + c1)
        p1 = c1 / (c0 + c1)

        #Gini impurity
        if (self.lossType == LossType.GINI):
            return (p0 * (1 - p0) + p1 * (1 - p1)) * (len(self.y) / n)

        #Cross entropy impurity
        if (self.lossType == LossType.CE):
            if (p0 == 0 or p1 == 0):
                return 0
            else:
                return (-0.5 * (p0 * np.log2(p0) + p1 * np.log2(p1))) * (len(self.y) / n)


def Construct_Subtree(node, max_depth, lossType):
    if (node.depth == max_depth or len(node.y) == 1):
        node.g = node.y.mean()

    else:
        j, xi = CalculateOptimalSplit(node, lossType)
        node.j = j
        node.xi = xi

        Xt, yt, Xf, yf = DataSplit(node.X, node.y, j, xi)

        if (len(yt) > 0):
            node.left = TNode(node.depth + 1, Xt, yt, lossType)
            Construct_Subtree(node.left, max_depth, lossType)

        if (len(yf) > 0):
            node.right = TNode(node.depth + 1, Xf, yf, lossType)
            Construct_Subtree(node.right, max_depth, lossType)

    return node

# split the data-set
def DataSplit(X, y, j, xi):
    ids = X[:, j] <= xi
    Xt = X[ids == True, :]
    Xf = X[ids == False, :]
    yt = y[ids == True]
    yf = y[ids == False]
    return Xt, yt, Xf, yf



def CalculateOptimalSplit(node, lossType):
    X = node.X
    y = node.y
    best_var = 0
    best_xi = X[0, best_var]
    best_split_val = node.CalculateLoss()

    m, n = X.shape

    for j in range(0, n):
        for i in range(0, m):
            xi = X[i, j]
            Xt, yt, Xf, yf = DataSplit(X, y, j, xi)
            tmpt = TNode(0, Xt, yt, lossType)
            tmpf = TNode(0, Xf, yf, lossType)
            loss_t = tmpt.CalculateLoss()
            loss_f = tmpf.CalculateLoss()
            curr_val = loss_t + loss_f
            if (curr_val < best_split_val):
                best_split_val = curr_val
                best_var = j
                best_xi = xi
    return best_var, best_xi

def Predict(X, node):
    if (node.right == None and node.left != None):
        return Predict(X, node.left)

    if (node.right != None and node.left == None):
        return Predict(X, node.right)

    if (node.right == None and node.left == None):
        return node.g
    else:
        if (X[node.j] <= node.xi):
            return Predict(X, node.left)
        else:
            return Predict(X, node.right)



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = makedata_clf()
    maxdepth = 10  # maximum tree depth

    n = len(X_train)

    # define loss type
    # lossType = LossType.GINI
    # lossType = LossType.CE
    lossType = LossType.MISSCLF

    # Create tree root at depth 0
    treeRoot = TNode(0, X_train, y_train, lossType)

    # Build the regression tree with maximal depth equal to max_depth
    Construct_Subtree(treeRoot, maxdepth, lossType)

    # Predict
    y_hat = np.zeros(len(X_test))
    for i in range(len(X_test)):
        y_hat[i] = Predict(X_test[i], treeRoot)

    print("Basic tree: tree loss = ", zero_one_loss(y_test, np.int64(y_hat)))


#%% Q2

# Q2 cross-validation for decision tree
from sklearn.datasets import make_blobs
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def custom_zer_one_score(model, X, y):
    y_pred = model.predict(X)
    return zero_one_loss(y, y_pred)

if __name__ == "__main__":
    X, y = make_blobs(n_samples=5000, n_features=10, centers=3,
                      random_state=10, cluster_std=10)

    model = DecisionTreeClassifier(random_state=0)

    tree_depth = range(1, 20)

    xlist = []
    trlist = []
    cvlist = []

    for d in tree_depth:
        xlist.append(d)
        model.max_depth = d
        cv = np.mean(cross_val_score(model, X, y, cv=10,
                                     scoring=custom_zer_one_score))
        cvlist.append(cv)
        model.fit(X, y)
        trlist.append(custom_zer_one_score(model, X, y))

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)

    f = plt.figure()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%5.2f'))

    cv = plt.plot(xlist, cvlist, '-s', linewidth=0.5,
                  label='cross-validation')
    tr = plt.plot(xlist, trlist, '-*', linewidth=0.5, label='train')
    plt.xlabel('tree depth', fontsize=18, color='black')
    plt.ylabel('loss', fontsize=18, color='black')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=14, loc=3)
    plt.show()


#%%Q4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

#read the data
data = pd.read_csv("train.csv")
data.head()

X = data.iloc[:, 1:]
y = data['label']
print(X)
print(y)

#%% (a) print image
import matplotlib.pyplot as plt
tmp = np.array(X.iloc[10].values.reshape((28,28))).astype(np.uint8)
img = Image.fromarray(tmp)
img.show()

#plt.imshow(img)
#plt.show()

# iloc is select by row number
X.iloc[10]

# select by column
X.iloc[:,10]



#%%(b) split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)


#%% (c)logistic regression model
reg = LogisticRegression(solver = "lbfgs") # solver identify the penalty
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("Logistic Regression 0/1 loss = ", zero_one_loss(y_pred, y_test))


#%%(d)random forest model
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
print("Random Forest 0/1 loss = ", zero_one_loss(y_pred, y_test))

