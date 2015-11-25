import numpy as np
from scipy import stats

class GibbsSampling:

    def __init__(self,A, B, C, D, E, F, p_B, p_C, p_A_B, p_D_BC, p_E_D, p_F_C):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.p_B = p_B
        self.p_C = p_C
        self.p_A_B = p_A_B
        self.p_D_BC = p_D_BC
        self.p_E_D = p_E_D
        self.p_F_C = p_F_C
        self.a = ''   # creates a null state for random variable A
        self.b = ''   # creates a null state for random variable B
        self.c = ''   # creates a null state for random variable C
        self.d = ''   # creates a null state for random variable D
        self.e = ''   # creates a null state for random variable E
        self.f = ''   # creates a null state for random variable F
    def get_VarTuple(self):
        """
            return a list of all variabels' states
        """
        return (A[self.a], B[self.b], C[self.c], D[self.d], E[self.e], F[self.f])

    def initialize(self, varList):
        """
            Input: varList = [f,e,d,c,b,a]
            Output: none
            Function: initialize all the variables
        """
        self.a =  varList .pop()
        self.b =  varList .pop()
        self.c =  varList .pop()
        self.d =  varList .pop()
        self.e =  varList .pop()
        self.f =  varList .pop()
            
    def buildProposalDistrion_A(self):
        """
            Input: none
            Output: proposal distribution for variabel A
            Function: calculate the proposal distribution for variabel A
            which is equle to P(A|B)/sum_A`{P(A`|B)}
        """
        upper = self.p_A_B[:,self.b]
        bottom =upper.sum(axis = 0)
        proposalDis_A = upper/bottom
        return proposalDis_A
    
    def samplingAndUpdate_A(self):
        xk = np.arange(0,len(self.A))
        pk = self.buildProposalDistrion_A()
        custm = stats.rv_discrete(name='custm', values=(xk, pk))      
        R = custm.rvs(size=1)# Return a row vector
        self.a =  R[0]# update variable a
        #print 'xk',xk
        #print 'pk:',pk
        #print 'A state index: ',R
        
    def buildProposalDistrion_B(self):
        """
            Input: none
            Output: proposal distribution for variabel B
            Function: calculate the proposal distribution for variabel B
            which is equle to P(B)P(A|B)P(D|B,C)/sum_B`{P(B`)P(A|B`)P(D|B`,C)}
        """
        upper = self.p_B * self.p_A_B[self.a,:] * self.p_D_BC[self.d,self.c,:]
        bottom =upper.sum(axis = 0)
        proposalDis_B = upper/bottom
        return proposalDis_B
    
    def samplingAndUpdate_B(self):
        xk = np.arange(0,len(self.B))
        pk = self.buildProposalDistrion_B()
        custm = stats.rv_discrete(name='custm', values=(xk, pk))      
        R = custm.rvs(size=1)# Return a row vector
        self.b =  R[0]# update variable a
        #print 'xk',xk
        #print 'pk:',pk
        #print 'B state index: ',R
        
    def buildProposalDistrion_C(self):
        """
            Input: none
            Output: proposal distribution for variabel C
            Function: calculate the proposal distribution for variabel C
            which is equle to P(C)P(D|B,C)P(F|C)/sum_C`{P(C)P(D|B,C)P(F|C)}
        """
        upper = self.p_C * self.p_D_BC[self.d,:,self.b]* self.p_F_C[self.f,:] 
        bottom =upper.sum(axis = 0)
        proposalDis_C = upper/bottom
        return proposalDis_C
    
    def samplingAndUpdate_C(self):
        xk = np.arange(0,len(self.C))
        pk = self.buildProposalDistrion_C()
        custm = stats.rv_discrete(name='custm', values=(xk, pk))      
        R = custm.rvs(size=1)# Return a row vector
        self.c =  R[0]# update variable a
        #print 'xk',xk
        #print 'pk:',pk
        #print 'C state index: ',R
        
    def buildProposalDistrion_D(self):
        """
            Input: none
            Output: proposal distribution for variabel D
            Function: calculate the proposal distribution for variabel D
            which is equle to P(D|B,C)P(E|D)/sum_D`{P(D|B,C)P(E|D)}
        """
        upper = self.p_D_BC[:,self.c,self.b] * self.p_E_D[self.e,:] 
        bottom =upper.sum(axis = 0)
        proposalDis_D = upper/bottom
        return proposalDis_D
    
    def samplingAndUpdate_D(self):
        xk = np.arange(0,len(self.D))
        pk = self.buildProposalDistrion_D()
        custm = stats.rv_discrete(name='custm', values=(xk, pk))      
        R = custm.rvs(size=1)# Return a row vector
        self.d =  R[0]# update variable a
        #print 'xk',xk
        #print 'pk:',pk
        #print 'D state index: ',R

    def buildProposalDistrion_E(self):
        """
            Input: none
            Output: proposal distribution for variabel E
            Function: calculate the proposal distribution for variabel A
            which is equle to P(E|D)/sum_E`{P(E`|D)}
        """
        upper = self.p_E_D[:,self.d]
        bottom =upper.sum(axis = 0)
        proposalDis_E = upper/bottom
        return proposalDis_E
    
    def samplingAndUpdate_E(self):
        xk = np.arange(0,len(self.E))
        pk = self.buildProposalDistrion_E()
        custm = stats.rv_discrete(name='custm', values=(xk, pk))      
        R = custm.rvs(size=1)# Return a row vector
        self.e =  R[0]# update variable a
        #print 'xk',xk
        #print 'pk:',pk
        #print 'E state index: ',R
        
    def buildProposalDistrion_F(self):
        """
            Input: none
            Output: proposal distribution for variabel F
            Function: calculate the proposal distribution for variabel F
            which is equle to P(F|C)/sum_F`{P(F`|C)}
        """
        upper = self.p_F_C[:,self.c]
        bottom =upper.sum(axis = 0)
        proposalDis_F = upper/bottom
        return proposalDis_F
    
    def samplingAndUpdate_F(self):
        xk = np.arange(0,len(self.F))
        pk = self.buildProposalDistrion_F()
        custm = stats.rv_discrete(name='custm', values=(xk, pk))      
        R = custm.rvs(size=1)# Return a row vector
        self.f =  R[0]# update variable a
        #print 'xk',xk
        #print 'pk:',pk
        #print 'F state index: ',R

 # 1) Set Varaibels' state list        
A = [False, True]
B = ['n', 'm', 's']
C = [False, True]
D = ['healthy', 'carrier', 'sick', 'recovering']
E = [False, True]
F = [False, True]

# 2) Set Known probabilities distribution
p_B = np.array([0.97, 0.01, 0.02])
p_C = np.array([0.7, 0.3])
p_A_B = np.array([[0.9,0.8,0.3],[0.1,0.2,0.7]])
# Load P(D|B,C)
# axis = 0 represent D (4 variabels)
# axis = 1 represent C (2 variabels)
# axis = 2 represent B (3 variabels)
p_D_BC = np.zeros((4,2,3))
p_D_BC[0] = np.array([[0.9, 0.8, 0.1],[0.3,0.4,0.01]])
p_D_BC[1] = np.array([[0.08, 0.17, 0.01],[0.05,0.05,0.01]])
p_D_BC[2] = np.array([[0.01,0.01,0.87],[0.05,0.15,0.97]])
p_D_BC[3] = np.array([[0.01,0.02,0.02],[0.6,0.4,0.01]])
## E.G. 2by2 matrix with ones = p_D_BC.sum(axis = 0) where first row with C = False
## and second row is with C = True
p_E_D = np.array([[0.99,0.99,0.4,0.9],[0.01,0.01,0.6,0.1]])
p_F_C = np.array([[0.99,0.2],[0.01,0.8]])

# 3) Start Gibbs sampling algorithm
Gibbs = GibbsSampling(A, B, C, D, E, F, p_B, p_C, p_A_B, p_D_BC, p_E_D, p_F_C)
# varList = [F,E,D,C,B,A]
varStateIndexList = [ 0, 0, 0, 0, None, 0]
# 3.1) initialize all variabels with initial state
Gibbs.initialize(varStateIndexList)
# 3.2) start sampling until mixing
p_ABCDEF_try = {}# Target distribution is joint distribution for all variabels P(A,B,C,D,E,F)
T = 1000
for i in xrange(1,T):
    Gibbs.samplingAndUpdate_B()
    Gibbs.samplingAndUpdate_A()
    Gibbs.samplingAndUpdate_C()
    Gibbs.samplingAndUpdate_D()
    Gibbs.samplingAndUpdate_E()
    Gibbs.samplingAndUpdate_F()
    key = Gibbs.get_VarTuple()
    if p_ABCDEF_try.get(key) == None:
        p_ABCDEF_try[key] = 1
    else:
        p_ABCDEF_try[key] = p_ABCDEF_try.get(key) + 1
print '\n-------------Repeat until mixing--------------------------------------'
print 'There are ',len(p_ABCDEF_try.keys()),' entries'
print 'and ',T,' sets of samples'
print 'Let us check the next 10 samples states whether they change or not'
print 'in order to see whether it get into stationary probability'
last10 = 10
for i in xrange(1,last10):
    Gibbs.samplingAndUpdate_B()
    Gibbs.samplingAndUpdate_A()
    Gibbs.samplingAndUpdate_C()
    Gibbs.samplingAndUpdate_D()
    Gibbs.samplingAndUpdate_E()
    Gibbs.samplingAndUpdate_F()
    print Gibbs.get_VarTuple()

# 3.3) Sampling sufficient samples such as M samples in order to approximate
#      Target distribution which is joint distribution for all variabels P(A,B,C,D,E,F)
p_ABCDEF = {}
M = 1000
for i in xrange(1,T):
    Gibbs.samplingAndUpdate_B()
    Gibbs.samplingAndUpdate_A()
    Gibbs.samplingAndUpdate_C()
    Gibbs.samplingAndUpdate_D()
    Gibbs.samplingAndUpdate_E()
    Gibbs.samplingAndUpdate_F()
    key = Gibbs.get_VarTuple()
    if p_ABCDEF.get(key) == None:
        p_ABCDEF[key] = 1
    else:
        p_ABCDEF[key] = p_ABCDEF.get(key) + 1
print '\n-------------Repeat until geting enough samples-----------------------'
print 'There are ',len(p_ABCDEF.keys()),' entries for estimating'
print 'and ',M,' sets of samples'
print '\n'

# 4. Estimate P(E), P(E|B=s), and P(E|B=s, C=True) with target distribution
# P(A,B,C,D,E,F)

# 4.1) P(E) 
E_False = 0
for key in p_ABCDEF.keys():
    print key
    if key[4] == False:
        E_False += p_ABCDEF.get(key)
p_E_False = (1.0*E_False)/M
p_E = np.array([p_E_False, 1-p_E_False])
print 'P(E)'
print 'False True'
print p_E
print '\n'

# 4.2) P(E|B=s)
E_False = 0
E_True = 0
for key in p_ABCDEF.keys():
    if key[1] == 's':
        if key[4] == False:
            E_False += p_ABCDEF.get(key)
        elif key[4] == True:
            E_True  += p_ABCDEF.get(key)
if E_True + E_False == 0:
    p_E_B_s = np.array([0, 0])
else:
    p_E_False_B_s = (1.0*E_False)/(E_True + E_False)
    p_E_B_s = np.array([p_E_False_B_s, 1-p_E_False_B_s])
print 'P(E|B=s)'
print 'False True'
print p_E_B_s
print'\n'

# 4.3) P(E|B=s,C=T)
E_False = 0
E_True = 0
for key in p_ABCDEF.keys():
    if key[1] == 's' and key[2] == True:
        if key[4] == False:
            E_False += p_ABCDEF.get(key)
        elif key[4] == True:
            E_True  += p_ABCDEF.get(key)
if E_True + E_False == 0:
    p_E_B_s_C_T = np.array([0, 0])
else:
    p_E_False_B_s_C_T  = (1.0*E_False)/(E_True + E_False)
    p_E_B_s_C_T = np.array([p_E_False_B_s_C_T , 1-p_E_False_B_s_C_T ])
print 'P(E|B=s,C=True)'
print 'False True'
print p_E_B_s_C_T 
print '\n'

