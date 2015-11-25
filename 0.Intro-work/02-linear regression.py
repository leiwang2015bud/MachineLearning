import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Part 1.Implement norm equation method to obtain linear regression on the dataset for
# the price of housing in Boston. In the linear regression, the price of house in 
# Boston will be predicted with several features. Besides, the biggest weight for
# certain feature will also be found, which indicates this feature impacts the price 
# of house 

# Part 2.Moreover, the comparision between regression 
# with regularization and regression without regularization would also introduced.

# Part 1
#%matplotlib inline
# 1.1 Load the dataset on the price of housing in Boston
# https://archive.ics.uci.edu/ml/datasets/Housing
names = ['medv', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
data = pd.read_csv('housing_scale.csv', header=None, names=names)
#print data.head()

# 1.2 Remove the feature called 'chas'
# Chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
data.drop('chas', axis=1, inplace=True)
#print data.shape

# Plot the median value of the property (vertical axis) versus the tax rate (horizontal axis).
x = data['tax']
y = data['medv']
plt.plot(x, y, 'g*')
plt.title('Boston house prices')
plt.xlabel('Tax rate (normalised)')
plt.ylabel('Median property value ($1000s)')


# Regression without Regularization by using the norm equation
## The normal equations method to find the nice theta(parameters)
def w_ml_unregularised(x, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

## Return an row vector
def phi_quadratic(x):
    """Compute phi(x) for a single training example using 
    quadratic basis function."""
    D = len(x)
    # Features are (1, {x_i}, {cross terms and squared terms})
    # Cross terms x_i x_j and squared terms x_i^2 can be taken 
    # from upper triangle of outer product of x with itself
    return np.hstack((1, x, np.outer(x, x)[np.triu_indices(D)]))

# Split dataset into training set and testing set
def split_data(data):
    """Randomly split data into two equal groups"""
    np.random.seed(1)#It can be called again to re-seed the generator. 
    N = len(data)# find the rows number(subject number) for data matrix
    idx = np.arange(N) # build N x 1 row narray,[0, 1, 2, ..., N]
    np.random.shuffle(idx) # re-allocate position for every elements
    train_idx = idx[:int(N/2)] # sub-split the first int(N/2) elements
    test_idx = idx[int(N/2):] # sub-split the rest elements

    X_train = data.loc[train_idx].drop('medv', axis=1)
    # extract the elements = data[rows][cols]
    t_train = data.loc[train_idx]['medv']
    X_test = data.loc[test_idx].drop('medv', axis=1)
    t_test = data.loc[test_idx]['medv']
    
    return X_train, t_train, X_test, t_test

# Define the cost function
def rmse(X_train, t_train, X_test, t_test, w):
    """Return the RMSE for training and test sets"""
    N_train = len(X_train)
    N_test = len(X_test)

    # Training set error
    t_train_pred = np.dot(X_train, w)
    rmse_train = np.linalg.norm(t_train_pred - t_train) / np.sqrt(N_train)

    # Test set error
    t_test_pred = np.dot(X_test, w)
    rmse_test = np.linalg.norm(t_test_pred - t_test) / np.sqrt(N_test)

    return rmse_train, rmse_test

X_train, t_train, X_test, t_test = split_data(data)
w_unreg = w_ml_unregularised(X_train, t_train) # find the nice theta
rmse(X_train, t_train, X_test, t_test, w_unreg) # return a tuple

# Find the feature with the biggest weight and plot two figures, one for the training set and one for the test set.
max_feature = np.argmax(np.abs(w_unreg))# give the index of the maximum element
t_train_pred = np.dot(X_train, w_unreg)
t_test_pred = np.dot(X_test, w_unreg)
fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(121)
# find the set of subjects with max feature: col matrix
ax.plot(X_train[[max_feature]], t_train, 'b.', label='true')
ax.plot(X_train[[max_feature]], t_train_pred, 'r.', label='predicted')
ax.set_title('Boston house prices - training set')
ax.set_xlabel(X_train.columns[max_feature])
ax.set_ylabel('Median property value ($1000s)')
ax.legend()

ax = fig.add_subplot(122)
ax.plot(X_test[[max_feature]], t_test, 'b.', label='true')
ax.plot(X_test[[max_feature]], t_test_pred, 'r.', label='predicted')
ax.set_title('Boston house prices - test set')
ax.set_xlabel(X_test.columns[max_feature])# find the name of this column 
print 'The most important feature :',X_test.columns[max_feature]
ax.set_ylabel('Median property value ($1000s)')
ax.legend()

# Part 2
# Regression with regularization
def w_ml_regularised(Phi, t, lamda):
    return np.dot(np.dot(np.linalg.inv(lamda * np.eye(Phi.shape[1]) + np.dot(Phi.T, Phi)), Phi.T), t)

w_reg = w_ml_regularised(X_train, t_train, 1.1)
max_feature = np.argmax(np.abs(w_reg))
t_train_pred = np.dot(X_train, w_reg)
t_test_pred = np.dot(X_test, w_reg)
fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(121)
ax.plot(X_train[[max_feature]], t_train, 'b.', label='true')
ax.plot(X_train[[max_feature]], t_train_pred, 'r.', label='predicted')
ax.set_title('Boston house prices - training set')
ax.set_xlabel(X_train.columns[max_feature])
ax.set_ylabel('Median property value ($1000s)')
ax.legend()

ax = fig.add_subplot(122)
ax.plot(X_test[[max_feature]], t_test, 'b.', label='true')
ax.plot(X_test[[max_feature]], t_test_pred, 'r.', label='predicted')
ax.set_title('Boston house prices - test set')
ax.set_xlabel(X_test.columns[max_feature])
ax.set_ylabel('Median property value ($1000s)')
ax.legend()

rmse(X_train, t_train, X_test, t_test, w_reg)