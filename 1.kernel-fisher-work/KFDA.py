import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
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
"""
Build Kenerl matrix
"""    
def gaussianKernel(x,z,sigma):
    """
    x: the column vector for each subject, where includes all features
    z: landmarks, the column vector for each subject, where includes all features
    sigma: a constant
    return a constant k, we could view it as the similarity of x and z
    """
    k = np.exp(-1 *np.power(LA.norm(x - z),2)/(2*np.power(sigma,2)) )
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
    k = np.power((np.inner(x,z)+c),p)
    return k
    
    
def getGaussianKernelMatrix(X,sigma):
    """
    X: the origianl feature matrix, where X \in R^{n \times m}
    sigma : a given constant
    return K: the gaussian kernel matrix, where K \in R^{m \times m}
    """
    n,m = X.shape
    K = np.zeros((m,m))
    for j in xrange(0, m):
        for i in xrange(0, m):
            xj = X[:,j]
            zi = X[:,i]
            kji = gaussianKernel(xj,zi,sigma)
            K[i,j] += kji
    return K

# Test
# a = array([[2,1],[1,0]])
# print getGaussianKernelMatrix(a,3)

def getInhomoKernelMatrix(X,c,p):
    """
    X: the origianl feature matrix, where X \in R^{n \times m}
    sigma : a given constant
    return K: the gaussian kernel matrix, where K \in R^{m \times m}
    """
    n,m = X.shape
    K =np.zeros((m,m))
    for j in xrange(0, m):
        for i in xrange(0, m):
            xj = X[:,j]
            zi = X[:,i]
            kji = inhomoKernel(xj,zi,c,p)
            K[i,j] += kji
    return K
#####################################################################################    

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
Training
"""
def getbam(X,y):
    """
    X is in R^{n \times m}
    """
    X = X.T
    y = y.T
#   y>0 class1
#   y<0 class2
    X_class1,X_class2 = splitXintoTwoClass(X,y)
    a = X_class1.shape[0]
    b = X_class2.shape[0]
    m = X.shape[0]
    return b,a,m

def getJandaVector(b,a):
    oneax1 =  np.ones( (a,1))
    onebx1 =  np.ones( (b,1))
    zerobxa = np.zeros((b,a))
    zeroaxb = zerobxa.T
    Ib = np.eye(b)
    Ia = np.eye(a)
    J2 = (1.0/np.sqrt(b))*(Ib - 1.0/b*np.dot(onebx1,onebx1.T ))
    J1 = (1.0/np.sqrt(a))*(Ia - 1.0/a*np.dot(oneax1,oneax1.T ))
    Jup =  np.append(J1,zeroaxb,axis = 1 )
    Jdown =  np.append(zerobxa,J2,axis = 1 )
    J = np.append(Jup,Jdown,axis = 0 )
    
    aVector1 = np.append(1.0/a*oneax1,np.zeros((b,1)),axis = 0 )
    aVector2 = np.append(np.zeros((a,1)),1.0/b*onebx1,axis = 0 )
    aVector = aVector1 + aVector2
    return J,aVector
  
def getA(lamda,I,J,K,aVector):
    invContent = lamda*I+np.dot(np.dot(J,K),J)
    JInver = np.dot(J,inv(invContent))
    JK = np.dot(J,K)
    IJInverJK = I - np.dot(JInver,JK)
    A = (1.0/lamda)*np.dot( IJInverJK,aVector)
    return A

def trainKFDAmodel(XkfdaTraining, YkfdaTraining, KTraining, lamda):
    b,a,m= getbam(XkfdaTraining,YkfdaTraining)
    I = np.eye(m)
    J,aVector = getJandaVector(b,a)
    A = getA(lamda,I,J,KTraining,aVector)
    onemx1 =  np.ones( (m,1))
    rescal = onemx1 + ((b-a)/m)*YkfdaTraining
#     y1 = 0.5*(YkfdaTraining+1)
#     y2 = y1-YkfdaTraining
#     B = diag(rescal)-(2*b/m*a)*dot(y1,y1.T)-(2*a/m*b)*dot(y2,y2.T)
#     A = dot(inv(dot(B,KTraining)+lamda*I),YkfdaTraining)
    threshold =  (0.25/(a*b))*np.dot(np.dot(A.T,KTraining),rescal)
    return A,threshold
"""
Testing
"""  
def testKFEAmodel(A,a0,KTesting):
    """
    a0: the threshold for decision boudry, array((1,m))
    we could define the cretia of decsion boudry as 0
    """
    print 'threshold is ',a0.min(),'~',a0.max() 
    D = np.dot(A.T,KTesting.T)
    print 'D: ',np.mean(D)
    print 'Min projection: ',D.min(), 'Max project:',D.max()
#     We want to make sure we find the right threoshold a0
    bins = np.linspace(D.min(), D.max(), 100)
    plt.hist(D.T, bins, alpha=0.5, label='Decision boundry')
    plt.legend(loc='upper right')
    plt.show()
    
    a0 = np.mean(D)
    pred=np.sign(D-a0)
#     pred = np.ones((1,D.shape[1]));
#     class2 = D < 0
#     pred[class2] = -1
    return pred
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

###########################################################################################################################
""" 
    Seperate  raw dataset equally into training set and testing set
"""
import random, math

def k_fold(data, myseed, k):
    # Shuffle input
    random.seed=myseed
    random.shuffle(data)

    # Compute partition size given input k
    len_part=int(math.ceil(len(data)/float(k)))

    # Create one partition per fold
    train={}
    test={}
    for ii in range(k):
        test[ii]  = data[ii*len_part:ii*len_part+len_part]
        train[ii] = [jj for jj in data if jj not in test]

    return test[0],test[1] 

# data = array([1,2,3,4])
# train, test = k_fold(data, 11109, 2)
# print train,test
    
def seperateRowDataSet(X,y):
    """
    Input
    X: raw feature matrix, X \in R^{m \times n}
    y: label column vector, y \in R^{m \times 1}
    
    Return
    X_training, \in R^{n \times m/2}
    y_training, \in R^{n \times m/2}
    X_testing, \in R^{n \times m/2}
    y_testing, \in R^{n \times m/2}
    """
    m = X.shape[0]
    index =np.arange(m)
#     np.random.shuffle(index)
#     indexSeperate= np.split(index, 2)
    indextrain, indexTest = k_fold(index, 11109, 2)
    
#     X_training_index_vector = indexSeperate[0]
#     y_training_index_vector = indexSeperate[0]
    X_training = X[indextrain]
    y_training = y[indextrain]
    
#     X_testing_index_vector = indexSeperate[1]
#     y_testing_index_vector = indexSeperate[1]
    X_testing = X[indexTest]
    y_testing = y[indexTest]
    
    return X_training.T,y_training.T,X_testing.T,y_testing.T
def getTestGaussianKernelMatrix(X,Z,sigma):
    """
    X: the origianl feature matrix, where X \in R^{n \times m}
    sigma : a given constant
    return K: the gaussian kernel matrix, where K \in R^{m \times m}
    """
    n,m = X.shape
    K = np.zeros((m,m))
    for j in xrange(0, m):
        for i in xrange(0, m):
            xi = X[:,i]
            zj = Z[:,j]
            kij = gaussianKernel(xi,zj,sigma)
            K[i,j] += kij
    return K

# Test
# a = array([[2,1],[1,0]])
# print getGaussianKernelMatrix(a,3)

def getTestInhomoKernelMatrix(X,Z,c,p):
    """
    X: the origianl feature matrix, where X \in R^{n \times m}
    sigma : a given constant
    return K: the gaussian kernel matrix, where K \in R^{m \times m}
    """
    n,m = X.shape
    K = np.zeros((m,m))
    for j in xrange(0, m):
        for i in xrange(0, m):
            xi = X[:,j]
            zj = Z[:,j]
            kij = inhomoKernel(xi,zj,c,p)
            K[i,j] += kij
    return K
# Test
# a = array([[2,1],[1,0]])
# print getInhomoKernelMatrix(a,1,2) 

def getTrainTestGaussianKernelMatrix(X_training,X_testing, sigma):
    KGaussian_training = getGaussianKernelMatrix(X_training,sigma)
    KGaussian_testing = getTestGaussianKernelMatrix(X_training,X_testing ,sigma)
    return  KGaussian_training,KGaussian_testing

def getTrainTestInhomoKernelMatrix(X_training,X_testing,c,p):
    Kinhomo_training = getInhomoKernelMatrix(X_training,c,p) 
    Kinhomo_testing = getTestInhomoKernelMatrix(X_training,X_testing,c,p) 
    return Kinhomo_training,Kinhomo_testing

def getNorKernelMatrix(Kraw_traing, Kraw_testing):
    ## Normalisation training kernel matrix
    KTraining_normlised = normalise_unit_diag(Kraw_traing)
    KTraining = KTraining_normlised
    
    ##Normalisation testing kernel matrix
    KTesting_normlised = normalise_unit_diag(Kraw_testing)
    KTesting = KTesting_normlised #The size of KTesting should be same as KTraining
    return KTraining, KTesting

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
    
def oneExperimentWithKFDA(X_training,y_training,y_testing,Kraw_traing, Kraw_testing,lamda):
    """
    We implement Gaussain kernel and center as well as normalise kernel matrix
    
    Input
    X_training, \in R^{n \times m/2}
    y_training, \in R^{n \times m/2}
    y_testing, \in R^{n \times m/2}
    Kraw_traing,  \in R^{m/2 \times m/2}
    Kraw_testing, \in R^{m/2 \times m/2}
    lamda :a positive regularization parameter
    a0: the threshold for decision boundry
    """
    KTraining, KTesting = getNorKernelMatrix(Kraw_traing, Kraw_testing)
#     KTraining, KTesting = getCenteringNorKernelMatrix(Kraw_traing, Kraw_testing)
    print KTraining
#     KTraining, KTesting = Kraw_traing, Kraw_testing

    ########################## Training  ######################################################################
    A,threshold = trainKFDAmodel(X_training, y_training, KTraining, lamda)
    
    ########################## Testing  ######################################################################
    predKFDA = testKFEAmodel(A,threshold,KTesting)
    
    ########################## Evaluating  ######################################################################
    M = confusion_matrix((np.hstack(predKFDA)+1)/2,(y_testing+1)/2)
    print M
    print accuracy(M)
    print balanced_accuracy(M)
    return balanced_accuracy(M)







####################################################################### Test ###########################################
lamda = 0.01
sigmaVector = np.array([0.23,1.1,8.7])
c = 1
pVector = np.array([2,3])
for i in xrange(0,1):
#     sigma = sigmaVector[i]
    sigma = 1.1

    for j in xrange(0,1):
        X_training,y_training,X_testing,y_testing = seperateRowDataSet(X,y);
        Kraw_traing, Kraw_testing = getTrainTestGaussianKernelMatrix(X_training,X_testing, sigma)
        print Kraw_traing
        ba = oneExperimentWithKFDA(X_training,y_training,y_testing,Kraw_traing, Kraw_testing,lamda)
        
# for i in xrange(0,2):
#     p = pVector[i]
#     for j in xrange(0,10):
#         X_training,y_training,X_testing,y_testing = seperateRowDataSet(X,y);
#         Kraw_traing, Kraw_testing = getTrainTestInhomoKernelMatrix(X_training,X_testing,c,p)
#         ba = oneExperimentWithKFDA(X_training,y_training,y_testing,Kraw_traing, Kraw_testing,lamda)
#         print ba 