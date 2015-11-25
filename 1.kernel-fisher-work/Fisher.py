import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import pandas as pd

raw_data = pd.read_csv('/Users/bud/GitHubProject/MachineLearning/1.kernel-fisher-work/census_abs2011_summary.csv')
#print(raw_data.shape)
raw_data.head()

# if age >= 38, labeled with 1
# else, labeled with -1.
labelvec = np.array(raw_data['Median_age_of_persons_Census_year_2011'])
y = np.ones(len(labelvec))
neg = labelvec < 38
y[neg] = -1 

# positive examples: larger than or equal to 38
# negative examples: smaller than 38
num_pos = len(np.flatnonzero(y > 0))
num_neg = len(np.flatnonzero(y < 0))
#print('Number of positive/negative examples = %d/%d' % (num_pos, num_neg))

headers = list(raw_data.columns.values) # get the features' name
headers.remove('Median_age_of_persons_Census_year_2011')
raw_feat = np.array(raw_data[headers]) # feature matrix without age feature  

avg = np.mean(raw_feat,axis = 0)
std_dev = np.std(raw_feat, axis = 0)


X = (raw_feat-avg)/std_dev # scaled features matrix [-1,1]
#print X.shape # X is m x n, where m is 342 subjects, n is 7 features
#print y.shape # y is m x 1, where m is 342 subjects

"""
Evaluation
"""
def confusion_matrix(prediction, labels):
    """Returns the confusion matrix for a list of predictions and (correct) labels
        prediction = [y_predict1, y_predict2, ..., y_predictm]
        labels = [y_1, y_2, ..., y_m]
        
        reuturn a matrix
        cmatrix = [[tp,fp],[tn,fn]] size: 2 x 2
    """
    assert len(prediction) == len(labels) # make sure there are same example numbers
    def f(pr, la):
        n = 0
        for i in range(len(prediction)):
            if prediction[i] == pr and labels[i] == la:
                n += 1
        return n
    return np.matrix([[f(1, 1), f(1, 0)], [f(0, 1), f(0, 0)]])

def accuracy(cmatrix):
    """Returns the accuracy of a confusion matrix
        accuracy = correct prediction number / all prediction number
    """
    tp, fp, fn, tn = cmatrix.flatten().tolist()[0]
    return 1.0*(tp + tn) / (tp + fp + fn + tn)

def balanced_accuracy(cmatrix):
    """Returns the balanced accuracy of a confusion matrix       
    """
    tp, fp, fn, tn = cmatrix.flatten().tolist()[0]
    return 1.0*tp / 2 / (tp + fn) + 0.5*tn / float(tn + fp)
def drawDistribution(X_projection1, X_projection2):
    """
    We would draw the projection distributions of two classes points onto
    the direction vector w to see wether they are in similar distribution.
    If they do, we could find the threshold by calculate the center of two means 
    of distributions, like w0 = 0.5*(mu_1 + mu_2).
    """
    plt.hist(X_projection1, bins=50, label='class 1(positive)')
    plt.hist(X_projection2, bins=50, label='class 2(negtive)')       # matplotlib version (plot)
    plt.legend(loc='upper right')
    plt.title('Projection onto the w')
    plt.xlabel('projection position')
    plt.ylabel('number of people in same projection position')
    plt.show()

def splitXintoTwoClass(X,y):
    """
    This function is to find the subjects in class1(y>0 positive), and in class2(y<=0 negtive)
    (narray,narray) ---> (narray, narray)
    """
    class1 = y>0
    class2 = y<0
    X_class1 =  X[class1]
    X_class2 = X[class2]
    return X_class1,X_class2
# TEST FOR FUNCTION splitXintoTwoClass(X,y)
# X_class1,X_class2 = splitXintoTwoClass(X,y)
# print 'TEST FOR FUNCTION splitXintoTwoClass(X,y): ',(len(X_class1.shape)==2)


#################################### Training ######################################
def getMeanAndVariance(X_classi):
    """
    This function would give us the mean and variance for each feature of a feature matrix
    (narray) ---> (narray, narray)
    """
    m_i = X_classi.mean(axis = 0) # we would calculate the mean of each column(feature) of X_classi
    sigma2_i = np.cov(X_classi.T) # we would calculate the corvarince of each column(feature) of X_classi
    N = X_classi.shape[0]
    sigma2_i = sigma2_i*(N-1) # FOLLOW THE LECTURE SLIDS
    return m_i,sigma2_i

# TEST FOR FUNCTION getMeanAndVariance(X_classi)
# X_classi = np.array([[1, 2], [3, 4]])
# m_i,sigma2_i = getMeanAndVariance(X_classi)
# print 'TEST FOR FUNCTION getMeanAndVariance(X_classi) m_i: ',m_i == array([[2.0,3.0]])
# print 'TEST FOR FUNCTION getMeanAndVariance(X_classi) sigma2_i: ',sigma2_i == array([[1.0, 1.0]])

def getDirectionVector(S_w, m_1, m_2):
    W = np.dot(inv(S_w),(m_2.T-m_1.T)) # because our feature matrix is the transpose format of a feature matrix in theory
    return W                    # we get W as W = (m2 - m1)S_w^{-1}, we cannot inverse the position of m_2 and m_1

def getMuAndProjection(W, X_classi):
    X_projection_i = np.dot(X_classi,W.T)
    mu_i = X_projection_i.mean()
    return mu_i,X_projection_i
    
def getThreshold(mu_1,mu_2,n1,n2,n):
    w0 = (1.0*n1/n)*mu_1 + (1.0*n2/n)*mu_2
    return w0
    
def train_fld(X,y):
#1) split X into two classes
    X_class1,X_class2 = splitXintoTwoClass(X,y)
#2) Calculate necessary matrix for each feature class
    m_1,sigma2_1 = getMeanAndVariance(X_class1)
    m_2,sigma2_2 = getMeanAndVariance(X_class2)
    S_w = sigma2_1 + sigma2_2
#3) get the direction vector
    W = 1* getDirectionVector(S_w, m_1, m_2)
#4) get the mean for each projection onto the direction vector
    mu_1,X_projection_1 = getMuAndProjection(W, X_class1)
    mu_2,X_projection_2 = getMuAndProjection(W, X_class2)
## check the distribution of histogram of projection for two classes
    drawDistribution(X_projection_1, X_projection_2)
#5) get the threshold
    w0 = getThreshold(mu_1,mu_2,X_class1.shape[0],X_class2.shape[0],X.shape[0])
    return W,w0


###################################### Testing ##################################

def predict_fld(W, w0,X):
    X_projection = np.dot(X,W.T)
    class2 = X_projection > w0
    pred = np.ones(len(X))
    pred[class2] = -1
#     pred = np.sign(X_projection - w0)

    return pred
    
XTraining = X
yTraining = y
W,w0 = train_fld(XTraining,yTraining)

XTesting = X
pred = predict_fld(W,w0,XTesting)

cmatrix = confusion_matrix((pred+1)/2,(y+1)/2)
print cmatrix
print accuracy(cmatrix),balanced_accuracy(cmatrix)
