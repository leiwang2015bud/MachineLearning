import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import pandas as pd

"""
    Goal: In this project, we use the census data from the Australian Bureau of 
    Statistics to do classification. For each example, it is an person.
    There are 7 processed features for each person, which includes
    a set of median and mean values for different regions of Australia, such as
    weekly rent, total family income, household size, and number of persons
    per bedroom. By using these 7 features we classify them into two groups, 
    whose the median age in a region is 38 or older. 
    
    Algorithm: Kernel fisher discriminant analysis
    
    Referrence: 
        https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis
        http://www.abs.gov.au
"""

'''
    Preparation for training and testing data set: 
    1) split data randomly into training and testing data set
    2) preprocess raw data

'''
def split_data(data,dropName):
    """Randomly split data into two equal groups"""
    # np.random.seed(1)#It can be called again to re-seed the generator. 
    N = len(data)# find the rows number(subject number) for data matrix
    idx = np.arange(N) # build N x 1 row narray,[0, 1, 2, ..., N]
    np.random.shuffle(idx) # re-allocate position for every elements
    train_idx = idx[:int(1.0*N/2)] # sub-split the first int(N/2) elements
    test_idx = idx[int(1.0*N/2):] # sub-split the rest elements

    X_train = data.loc[train_idx].drop(dropName, axis=1)
    # extract the elements = data[rows][cols]
    t_train = data.loc[train_idx][dropName]

    X_test = data.loc[test_idx].drop(dropName, axis=1)
    t_test = data.loc[test_idx][dropName]
    
    return X_train, t_train, X_test, t_test

def preprocessRawData(raw_data,y_raw):
    # if age >= 38, labeled with 1
    # else, labeled with -1.
    labelvec = np.array(y_raw)
    y = np.ones(len(labelvec))
    neg = labelvec < 38
    y[neg] = -1 

    # positive examples: larger than or equal to 38
    # negative examples: smaller than 38
    num_pos = len(np.flatnonzero(y > 0))
    num_neg = len(np.flatnonzero(y < 0))
    #print('Number of positive/negative examples = %d/%d' % (num_pos, num_neg))

    headers = list(raw_data.columns.values) # get the features' name
    raw_feat = np.array(raw_data[headers]) # feature matrix without age feature  

    avg = np.mean(raw_feat,axis = 0)
    std_dev = np.std(raw_feat, axis = 0)


    X = (raw_feat-avg)/std_dev # scaled features matrix [-1,1]
    #print X.shape # X is N x n, where N is 342 subjects, n is 7 features
    #print y.shape # y is N x 1, where N is 342 subjects
    return X,y

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

'''
    Evaluation: 
    1) Accuracy
    2) Banlanced Accuracy

'''
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

def confusion_matrix_advanced(prediction, labels):
    """Returns the confusion matrix for a list of predictions and (correct) labels"""
    assert len(prediction) == len(labels)
    f = lambda p, l: len(list(filter(lambda x: x == (p, l), zip(prediction, labels))))
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
    return 0.5*tp / float(tp + fn) + 0.5*tn / float(tn + fp)

'''
    Kernel Fisher discriminant analysis Algorithm
    1) Define the kernel and kernel matrix
    2) Training: Express the kernel fisher discriminant analysis equation
'''

'''
    Kernel 
    1) Gaussian kernel
    2) Inhomo Kernel

'''
def gaussianKernel(x,z,sigma):
    """
    x: the column vector for each subject, where includes all features
    z: landmarks, the column vector for each subject, where includes all features
    sigma: a constant
    return a constant k, we could view it as the similarity of x and z
    """
    k = np.exp(-1.0 *np.power(LA.norm(x - z),2)/(2.0*np.power(sigma,2)) )
    # k = np.exp(-1.0*np.dot(x.T,x)/(2.0*np.power(sigma,2)))*np.exp(np.dot(x.T,z)/(np.power(sigma,2)))*np.exp(-1.0*np.dot(z.T,z)/(2.0*np.power(sigma,2)))
    return k

#TEST
# a = np.array([[2,1]])
# b = np.array([[1,0]])
# print a.T,'\n', b.T
# print np.power(LA.norm(a-b),2)
# print 2*np.power(3,2)
# print np.exp(-2.0/18)
# print gaussianKernel(a.T,b.T,3)

def inhomoKernel(x,z,c,p):
    """
    x: the column vector for each subject, where includes all features
    z: landmarks, the column vector for each subject, where includes all features
    c,p : a constant
    return a constant from k, where k array only have one element
    
    """
    k = np.power((np.inner(x,z)+c*1.0),p)
    return k

#TEST
# a = np.array([[2,1]])
# b = np.array([[1,0]])
# print a,'\n', b
# print 'kernel',inhomoKernel(a,b,1,2) 

'''
    Kernel Kernel Fisher discriminant analysis
    1) Via Gaussian Kernel
    2) Via Inhomo Kernel
'''

def getMijByInhomoKernel(X_classi,X,c,p):
    l_i,n = X_classi.shape
    m,n = X.shape
    M_i = np.zeros((m,1))
    for j in xrange(0,m):
        x_j = X[j,:]
        for k in xrange(0,l_i):
            x_k_i = X_classi[k,:]
            M_i[j] += (1.0/l_i)*inhomoKernel(x_j,x_k_i,c,p)
    return M_i

def getNjByInhomoKernel(X_classi,X,c,p):
    l_i,n = X_classi.shape
    m,n = X.shape
    K_j = np.zeros((m,l_i))
    for i in xrange(0,m):
        x_n = X[i,:]
        for j in xrange(0,l_i):
            x_m = X_classi[j,:]
            K_j[i,j] += inhomoKernel(x_n,x_m,c,p)
    one_lix1 = (1.0/l_i)*np.ones((l_i,l_i))
    I = np.eye(l_i)
    meddle = I - one_lix1
    left = np.dot(K_j, meddle)
    right = K_j.T
    N_j = np.dot(left,right)
    return N_j

def getPredByInhomoKernel(alpha,X_new,X,c,p):
    l_i,n = X_new.shape
    m,n = X.shape
    K_i = np.zeros((m,l_i))
    for j in xrange(0,m):
        x_i = X[j,:]
        for k in xrange(0,l_i):
            x = X_new[k,:]
            K_i[j,k] += inhomoKernel(x_i,x,c,p)
    y_i = np.dot(alpha.T,K_i)
    return y_i
def trainKFDAwithInhomoKernel(X_train,y_train,X_test,c,p):
    X_class1,X_class2 = splitXintoTwoClass(X_train,y_train)

    M_1 = getMijByInhomoKernel(X_class1,X_test,c,p)
    M_2 = getMijByInhomoKernel(X_class2,X_test,c,p)

    N_1 = getNjByInhomoKernel(X_class1,X_test,c,p)
    N_2 = getNjByInhomoKernel(X_class2,X_test,c,p)
    N = N_1 + N_2

    alpha = getDirectionVector(N, M_1, M_2)
    y_pred = getPredByInhomoKernel(alpha,X_test,X_train,c,p)

    return y_pred






def getMijByGaussianKernel(X_classi,X,sigma):
    l_i,n = X_classi.shape
    m,n = X.shape
    M_i = np.zeros((m,1))
    for j in xrange(0,m):
        x_j = X[j,:]
        temp = 0
        for k in xrange(0,l_i):
            x_k_i = X_classi[k,:]
            temp += gaussianKernel(x_j,x_k_i,sigma)
        M_i[j] = (1.0/l_i)*temp
    return M_i

def getNjByGaussianKernel(X_classi,X,sigma):
    l_i,n = X_classi.shape
    m,n = X.shape
    K_j = np.zeros((m,l_i))
    for i in xrange(0,m):
        x_n = X[i,:]
        for j in xrange(0,l_i):
            x_m = X_classi[j,:]
            K_j[i,j] += gaussianKernel(x_n,x_m,sigma)
    one_lix1 = (1.0/l_i)*np.ones((l_i,l_i))
    I = np.eye(l_i)
    meddle = I - one_lix1
    left = np.dot(K_j, meddle)
    right = K_j.T
    N_j = np.dot(left,right)
    return N_j

def getDirectionVector(N, M_1, M_2):
    alpha = np.dot(inv(N),(M_2-M_1)) 
    return alpha                    

def getPredByGaussianKernel(alpha,X_new,X,sigma):
    l_i,n = X_new.shape
    m,n = X.shape
    K_i = np.zeros((m,l_i))
    for j in xrange(0,m):
        x_j = X[j,:]
        for k in xrange(0,l_i):
            x = X_new[k,:]
            K_i[j,k] = gaussianKernel(x_j,x,sigma)
    y_i = np.dot(alpha.T,K_i)
    return y_i

def trainKFDAwithGaussianKernel(X_train,y_train,X_test,sigma):
    X_class1,X_class2 = splitXintoTwoClass(X_train,y_train)
    n1, col = X_class1.shape
    n2, col = X_class2.shape
    n = n1+n2

    M_1 = getMijByGaussianKernel(X_class1,X_train,sigma)
    M_2 = getMijByGaussianKernel(X_class2,X_train,sigma)

    N_1 = getNjByGaussianKernel(X_class1,X_train,sigma)
    N_2 = getNjByGaussianKernel(X_class2,X_train,sigma)
    N = N_1 + N_2 + 0.1*np.eye(n)# We have to add this in case of the singularity of N
    # N = N_1 + N_2

    alpha = getDirectionVector(N, M_1, M_2)
    y = getPredByGaussianKernel(alpha,X_test,X_train,sigma)

    dist1 = np.fabs(y - np.dot(alpha.T,M_1))
    dist2 = np.fabs(y - np.dot(alpha.T,M_2))
    dist1 = dist1.flatten()
    dist2 = dist2.flatten()

    y_pred = np.ones(len(X_test))
    class2=np.asarray(dist1 > dist2) 
    y_pred[class2] = -1

    return y_pred


'''
1. Load data and split them into feature matrix and target vector
    y: colum target vector, m x 1, where m is 342 subjects
    X: feature matrix, m x n, where m is 342 subjects, n is 7 features
'''
raw_data = pd.read_csv('/Users/bud/GitHubProject/MachineLearning/1.kernel-fisher-work/census_abs2011_summary.csv')
#print(raw_data.shape)
#print raw_data.head()

dropName = 'Median_age_of_persons_Census_year_2011'
X_train_raw, y_train_raw, X_test_raw, y_test_raw = split_data(raw_data,dropName)
X_train,y_train = preprocessRawData(X_train_raw,y_train_raw)
X_test,y_test = preprocessRawData(X_test_raw,y_test_raw)

# '''
# 2. Start training and testing 

# '''
# # 1) With Gaussian kernel
sigma = 1.1
y_pred = trainKFDAwithGaussianKernel(X_train,y_train,X_test,sigma)

M = confusion_matrix((y_pred+1)/2,(y_test+1)/2)
# M = confusion_matrix(y_pred,(y_test+1)/2)

print 'Confusion_matrix:\n',M
# print accuracy(M),':Accuracy'
print balanced_accuracy(M),':Banlanced accuracy'
print 'hi'








    
