import numpy as np

X1 = np.array([[2,1],[1,2],[3,4]])
one22 = np.ones((2,2))
print 1.0*np.dot(X1,X1.T)-1.0/2
print 1.0*np.dot(np.dot(X1,one22),X1.T)/2

X2 = np.array([[5,7,7],[8,33,10],[10,11,22]])
one23 = np.ones((2,3))
one32 = np.ones((3,2))
one33 = np.ones((3,3))

#print 1.0*np.dot(X2,X2.T)-2
#print 1.0*np.dot(np.dot(X2,one33),X2.T)/3

X1X2 =  np.dot(np.dot(X1,np.dot(X1.T,X2)),X2.T)
X11X2= np.dot(np.dot(X1,one23),X2.T)/6
#print X1X2/np.linalg.norm(X1X2)
#print X11X2/np.linalg.norm(X11X2)

#print np.dot(one22,one22)
one21 = np.ones((2,1))
one11 = np.ones((1,1))
#print np.dot(X1,one22)

w = np.array([[2,1,2]]).T
#print np.dot(np.dot(np.dot(w.T,X1),one21)/2,one21.T)