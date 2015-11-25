import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Load raw data set
raw_data = pd.read_csv('/Users/bud/GitHubProject/MachineLearning/kernel-fisher-work/census_abs2011_summary.csv')
print(raw_data.shape)
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
print('Number of positive/negative examples = %d/%d' % (num_pos, num_neg))

headers = list(raw_data.columns.values) # get the features' name
headers.remove('Median_age_of_persons_Census_year_2011')
raw_feat = np.array(raw_data[headers]) # feature matrix without age feature  

avg = np.mean(raw_feat,axis = 0)
std_dev = np.std(raw_feat, axis = 0)


X = (raw_feat-avg)/std_dev # scaled features matrix [-1,1]
print X.shape # X is m x n, where m is 342 subjects, n is 7 features
print y.shape # y is m x 1, where m is 342 subjects

# Solution goes here
#################################### Training ######################################
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

def getMeanAndVariance(X_classi):
    """
    This function would give us the mean and variance for each feature of a feature matrix
    (narray) ---> (narray, narray)
    """
    m_i = X_classi.mean(axis = 0) # we would calculate the mean of each column(feature) of X_classi
    sigma2_i = np.cov(X_classi.T) # we would calculate the mean of each column(feature) of X_classi
    N = X_classi.shape[0]
    sigma2_i = sigma2_i*(N-1) # FOLLOW THE LECTURE SLIDS
    return m_i,sigma2_i
# TEST FOR FUNCTION getMeanAndVariance(X_classi)
# X_classi = np.array([[1, 2], [3, 4]])
# m_i,sigma2_i = getMeanAndVariance(X_classi)
# print 'TEST FOR FUNCTION getMeanAndVariance(X_classi) m_i: ',m_i == array([[2.0,3.0]])
# print 'TEST FOR FUNCTION getMeanAndVariance(X_classi) sigma2_i: ',sigma2_i == array([[1.0, 1.0]])

def getDirectionVector(S_w, m_1, m_2):
    W = np.dot((m_1-m_2),np.linalg.inv(S_w)) # because our feature matrix is the transpose format of a feature matrix in theory
    return W                    # we get W as W = (m2 - m1)S_w^{-1}, we cannot inverse the position of m_2 and m_1

def getMuAndProjection(W, X_classi):
    X_projection_i = np.dot(X_classi,W.T)
    mu_i = X_projection_i.mean()
    return mu_i,X_projection_i
    
def getThreshold(mu_1,mu_2):
    w0 = 0.5*(mu_1 + mu_2)
    return w0
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
    return 1.0*tp / 2 / (tp + fn) + tn / 2 / (tn + fp)
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
    w0 = getThreshold(mu_1,mu_2)
    return W,w0


###################################### Testing ##################################
def predict_fld(W, w0,X):
    X_projection = np.dot(W,X.T)
#     class2 = X_projection < w0
#     pred = np.ones(len(X))
#     pred[class2] = -1
    pred = np.sign(X_projection - w0)
    return pred
    
XTraining = X
yTraining = y
W,w0 = train_fld(XTraining,yTraining)
#print w0

XTesting = X
pred = predict_fld(W,w0,XTesting)
# print pred

cmatrix = confusion_matrix((pred+1)/2,(y+1)/2)
print cmatrix
print accuracy(cmatrix),balanced_accuracy(cmatrix)




"""
Training
"""
def getbam(X,y):
    """
    X is in R^{m \times n}
    y is in R^{m \times 1}
    """
#   y>0 class1
#   y<0 class2
    X_class1,X_class2 = splitXintoTwoClass(X,y)
    a = X_class1.shape[0]
    b = X_class2.shape[0]
    m = X.shape[0]
    return b,a,m

def trainKFDAmodel(XkfdaTraining, YkfdaTraining, KTraining, lamda):
    b,a,m= getbam(XkfdaTraining,YkfdaTraining)
    I = np.eye(m)
    onemx1 =  np.ones( (m,1))
    y1 = 0.5*(YkfdaTraining+1)
    y2 = y1-YkfdaTraining
    rescal = onemx1 + (1.0*(b-a)/m)*YkfdaTraining
    B = np.diag(rescal)-(2.0*b/(m*a))*np.dot(y1,y1.T)-(2.0*a/(m*b))*np.dot(y2,y2.T)
    A = np.dot(np.linalg.inv(np.dot(B,KTraining)+lamda*I),YkfdaTraining)
    threshold =  (0.25/(a*b))*np.dot(np.dot(A.T,KTraining),rescal)
    return A,threshold
"""
Testing
"""  
def testKFEAmodel(A,a0,KTesting):
    """
    a0: the threshold for decision boudry, array((m,1))
    we could define the cretia of decsion boudry as 0
    """
    D = np.dot(KTesting.T,A)-a0
    pred=np.sign(D)
#     print 'Min projection: ',D.min(), 'Max project:',D.max()
#     We want to make sure we find the right threoshold a0
#     bins = numpy.linspace(D.min(), D.max(), 100)
#     pyplot.hist(D.T, bins, alpha=0.5, label='Decision boundry')
#     pyplot.legend(loc='upper right')
#     pyplot.show()
    return pred


"""
Centering and Normalisation of kernel
"""
# Centering
def center(K):
    """Center the kernel matrix, such that the mean (in feature space) is zero."""
    n = K.shape[0]
    one_nx1 = np.ones((n,1))
    one_nxn = np.ones((n,n))
    D = K.mean(axis = 0)
    J = one_nx1*D
    E = D.mean()*one_nxn
    C = K - J -J.T +E   
    return C

#Normalising
def normalise_unit_diag(K):
    """Normalise the kernel matrix, such that all diagonal entries are 1."""
    D_vector = 1.0/np.sqrt(np.diag(K)) 
    D = np.diag(D_vector, k = 0)
    R = np.dot(D,np.dot(K,D))
    return R
# #Testing
# a = array([[1,2,3],[4,5,6],[7,8,9]])
# R = normalise_unit_diag(a)
# print R
    
"""
    Seperation:  raw dataset equally into training set and testing set
"""
def seperateRowDataSet(X,y):
    """
    Input
    X: raw feature matrix, X \in R^{m \times n}
    y: label column vector, y \in R^{m \times 1}
    
    Return
    X_training, \in R^{m/2 \times n}
    y_training, \in R^{m/2 \times 1}
    X_testing, \in R^{m/2 \times n}
    y_testing, \in R^{m/2 \times 1}
    """
    X_class1,X_class2 = splitXintoTwoClass(X,y)
    a = X_class1.shape[0]
    b = X_class2.shape[0]    
    idxa = np.arange(a)
    idxb = np.arange(b)   
    np.random.seed=11109
    np.random.shuffle(idxa)
    np.random.seed=11109
    np.random.shuffle(idxb)
    ############### class 1 
    am = 1+((a-1)/2)
    train_idx_a = idxa[0:int(am)]
    test_idx_a = idxa[int(am):]
   
    X_training_a = X_class1[train_idx_a]
    y_training_a = np.ones((am,1))
    
    X_testing_a = X_class1[test_idx_a]
    y_testing_a = np.ones((((a-1)/2),1))    
    ################ class 2
    bm = 1+((b-1)/2)
    test_idx_b = idxb[0:int(bm)]
    train_idx_b = idxb[int(bm):]
   
    X_testing_b = X_class1[test_idx_b]
    y_testing_b = -1*np.ones((bm,1))
    
    X_training_b = X_class1[train_idx_b]
    y_training_b = -1*np.ones((((b-1)/2),1))
    ################ Combine together
    X_training = np.append(X_training_a,X_training_b,axis = 0)
    y_training = np.append(y_training_a,y_training_b,axis = 0)
    
    X_testing = np.append(X_testing_a,X_testing_b,axis = 0)
    y_testing = np.append(y_testing_a,y_testing_b,axis = 0)
       
    return X_training,y_training,X_testing,y_testing
# import random, math

# def k_fold(data, myseed, k):
#     # Shuffle input
#     random.seed=myseed
#     random.shuffle(data)
#     # Compute partition size given input k
#     len_part=int(math.ceil(len(data)/float(k)))
#     # Create one partition per fold
#     train={}
#     test={}
#     for ii in range(k):
#         test[ii]  = data[ii*len_part:ii*len_part+len_part]
#         train[ii] = [jj for jj in data if jj not in test]

#     return test[0],test[1] 

# # data = array([1,2,3,4])
# # train, test = k_fold(data, 11109, 2)
# # print train,test
    
# def seperateRowDataSet(X,y):
#     """
#     Input
#     X: raw feature matrix, X \in R^{m \times n}
#     y: label column vector, y \in R^{m \times 1}
    
#     Return
#     X_training, \in R^{m/2 \times n}
#     y_training, \in R^{m/2 \times 1}
#     X_testing, \in R^{m/2 \times n}
#     y_testing, \in R^{m/2 \times 1}
#     """
#     m = X.shape[0]
#     index =np.arange(m)
#     indextrain, indexTest = k_fold(index, 11109, 2)
#     X_training = X[indextrain]
#     y_training = y[indextrain]
#     X_testing = X[indexTest]
#     y_testing = y[indexTest]    
#     return X_training,y_training,X_testing,y_testing

def gaussianKernel(x,z,sigma):
    """
    x: the column vector for each subject, where includes all features
    z: landmarks, the column vector for each subject, where includes all features
    sigma: a constant
    return a constant k, we could view it as the similarity of x and z
    """
    k = np.exp(-1.0 *np.power(np.linalg.norm(x - z),2)/(2.0*np.power(sigma,2)) )    
    return k

#TEST
# a = array([[2,1]])
# b = array([[1,0]])
# print a.T,'\n', b.T
# print power(norm(a-b),2)
# print power(3,2)
# print gaussianKernel(a,b,3)

def inhomoKernel(x,z,c,p):
    """
    x: the column vector for each subject, where includes all features
    z: landmarks, the column vector for each subject, where includes all features
    c,p : a constant
    return a constant from k, where k array only have one element
    
    """
    k = np.power((np.inner(x,z)+c*1.0),p)
    return k
"""
    Kernel matrix construction:
    1)Training kernel matrix
    2)Testing kernel matrix
"""
def getGaussianKernelMatrix(X,sigma):
    """
    X: the origianl feature matrix, where X \in R^{m \times n}
    sigma : a given constant
    return K: the gaussian kernel matrix, where K \in R^{m \times m}
    """
    m,n = X.shape
    K = np.zeros((m,m))
    for j in xrange(0, m):
        for i in xrange(0, m):
            xi = X[i,:]
            xj = X[j,:]
            kij = gaussianKernel(xi,xj,sigma)
            K[i,j] += kij
    return K

# Test
# a = array([[2,1],[1,0]])
# print getGaussianKernelMatrix(a,3)

def getInhomoKernelMatrix(X,c,p):
    """
    X: the origianl feature matrix, where X \in R^{m \times n}
    sigma : a given constant
    return K: the gaussian kernel matrix, where K \in R^{m \times m}
    """
    m,n = X.shape
    K = np.zeros((m,m))
    for j in xrange(0, m):
        for i in xrange(0, m):
            xi = X[i,:]
            xj = X[j,:]
            kij = inhomoKernel(xi,xj,c,p)
            K[i,j] += kij
    return K
# Test
# a = array([[2,1],[1,0]])
# print getInhomoKernelMatrix(a,1,2)

def getTestGaussianKernelMatrix(X,Z,sigma):
    """
    X: the origianl feature matrix, where X \in R^{m \times n}
    sigma : a given constant
    return K: the gaussian kernel matrix, where K \in R^{m \times m}
    """
    m,n = X.shape
    K = np.zeros((m,m))
    for j in xrange(0, m):
        for i in xrange(0, m):
            xi = X[i,:]
            zj = Z[j,:]
            kij = gaussianKernel(zj,xi,sigma)
            K[i,j] += kij
    return K

# Test
# a = array([[2,1],[1,0]])
# print getGaussianKernelMatrix(a,3)

def getTestInhomoKernelMatrix(X,Z,c,p):
    """
    X: the origianl feature matrix, where X \in R^{m \times n}
    sigma : a given constant
    return K: the gaussian kernel matrix, where K \in R^{m \times m}
    """
    m,n = X.shape
    K = np.zeros((m,m))
    for j in xrange(0, m):
        for i in xrange(0, m):
            xi = X[i,:]
            zj = Z[j,:]
            kij = inhomoKernel(zj,xi,c,p)
            K[i,j] += kij
    return K
# Test
# a = array([[2,1],[1,0]])
# print getInhomoKernelMatrix(a,1,2) 

def getTrainTestGaussianKernelMatrix(X_training,X_testing, sigma):
    KGaussian_training = getGaussianKernelMatrix(X_training,sigma)
    KGaussian_testing = getTestGaussianKernelMatrix(X_training,X_testing,sigma)
    return  KGaussian_training,KGaussian_testing

def getTrainTestInhomoKernelMatrix(X_training,X_testing,c,p):
    Kinhomo_training = getInhomoKernelMatrix(X_training,c,p) 
    Kinhomo_testing = getTestInhomoKernelMatrix(X_training,X_testing,c,p) 
    return Kinhomo_training,Kinhomo_testing

"""
    Normalisation of Kernel Matrix
"""
def getNorKernelMatrix(Kraw_traing, Kraw_testing):
    ## Normalisation training kernel matrix
    KTraining_normlised = normalise_unit_diag(Kraw_traing)
    KTraining = KTraining_normlised    
    ##Normalisation testing kernel matrix
    KTesting_normlised = normalise_unit_diag(Kraw_testing)
    KTesting = KTesting_normlised #The size of KTesting should be same as KTraining
    return KTraining, KTesting
#     return KTraining, Kraw_testing

def getCenteringNorKernelMatrix(Kraw_traing, Kraw_testing):
    ## Centering training kernel matrix
    KTraining_centered = center(Kraw_traing)
    ## Normalisation training kernel matrix
    KTraining_normlised = normalise_unit_diag(KTraining_centered)
    KTraining = KTraining_normlised    
    ##Centering testing kernel matrix
    KTesting_centered = center(Kraw_testing)
    ##Normalisation testing kernel matrix
    KTesting_normlised = normalise_unit_diag(KTesting_centered)
    KTesting = KTesting_normlised #The size of KTesting should be same as KTraining
    return KTraining, KTesting
#     return KTraining, Kraw_testing

"""
    Experiment Conduction
"""
def oneExperimentWithKFDA(X_training,y_training,y_testing,Kraw_traing, Kraw_testing,lamda):
    """
    We implement Gaussain kernel and center as well as normalise kernel matrix    
    Input:
    X_training, \in R^{n \times m/2}
    y_training, \in R^{n \times m/2}
    y_testing, \in R^{n \times m/2}
    Kraw_traing,  \in R^{m/2 \times m/2}
    Kraw_testing, \in R^{m/2 \times m/2}
    lamda :a positive regularization parameter
    Output:
    balanced_accuracy
    """
#     KTraining, KTesting = getNorKernelMatrix(Kraw_traing, Kraw_testing)
#     KTraining, KTesting = getCenteringNorKernelMatrix(Kraw_traing, Kraw_testing)
    KTraining, KTesting = Kraw_traing, Kraw_testing
    ########################## Training  ###################################################
    A,threshold = trainKFDAmodel(X_training, y_training, KTraining.T, lamda)    
    ########################## Testing  ####################################################
    predKFDA = testKFEAmodel(A,threshold,KTesting)   
    ########################## Evaluating  #################################################
    M = confusion_matrix((predKFDA+1)/2,(y_testing+1)/2)
    print M
    print accuracy(M)
#     print balanced_accuracy(M)
    return balanced_accuracy(M)
    
# Test
lamda = 0.00000001
sigmaVector = np.array([0.23,1.1,8.7])
c = 1
pVector = np.array([2,3])
for i in xrange(0,1):
    #sigma = sigmaVector[i]
    sigma = 8.7
    for j in xrange(0,10):
        X_training,y_training,X_testing,y_testing = seperateRowDataSet(X,y);
#         print y_training,y_testing
        Kraw_traing, Kraw_testing = getTrainTestGaussianKernelMatrix(X_training,X_testing, sigma)
        ba = oneExperimentWithKFDA(X_training,y_training,y_testing,Kraw_traing, Kraw_testing,lamda)
        print ba


#XTraining,yTraining,XTesting,yTesting = seperateRowDataSet(X,y);
#print yTraining[:,0].shape
#W,w0 = train_fld(XTraining,yTraining[:,0])
##print w0
#
#pred = predict_fld(W,w0,XTesting)
## print pred
#
#cmatrix = confusion_matrix((pred+1)/2,(yTesting[:,0]+1)/2)
#print cmatrix
#print accuracy(cmatrix),balanced_accuracy(cmatrix)