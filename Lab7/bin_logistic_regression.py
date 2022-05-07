import numpy
import sklearn.datasets
import scipy.optimize

'''
We will implement the binary
version of the logistic regression to discriminate between iris virginica and iris versicolor. We will ignore
iris setosa. We will represent labels with 1 (iris versicolor) and 0 (iris virginica).
'''

def vcol(v): #v is a row vector(1, v.size), the function returns a column vector (v.size,1)
    return v.reshape((v.size, 1))

def vrow(v): #v is a column vector(v.size,1), the function returns a row vector (1,v.size)
    return v.reshape((1,v.size))

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def logreg_obj_wrap2(DTR, LTR, l):
    def logreg_obj(v):
        M = DTR.shape[0]
        w,b = vcol(v[0:M]), v[-1]
        j = l/2 * numpy.linalg.norm(w)**2
        arg = 0.0
        for i in range(DTR.shape[1]): #
            x = DTR[:, i:i+1] #takes one column at a time
            z = 2*LTR[i] - 1
            arg += numpy.logaddexp(0, -z*(numpy.dot(w.T, x)+b)) 
        return j + 1/DTR.shape[1]*arg
    return logreg_obj

def correct_predictions(predicted_labels, LTE):
    correct = 0
    for i,x in enumerate(predicted_labels):
        if x == LTE[i]:
            correct += 1
    return correct

'''def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR*2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v):
        w,b = vcol(v[0:M]), v[-1]
        S = numpy.dot(w.T, DTR) + b     #scores for all training set (or i can do it using for loop)
        xce = numpy.logaddexp(0,-Z*S).mean()   #log(1 + exp..) = logaddexp(0, -zi(si))
        return xce + 0.5*l + numpy.linalg.norm(w)**2
    return logreg_obj
'''

if __name__ == '__main__':

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    for l in [1e-6,1e-3,0.1,1.0]:
        logreg_obj = logreg_obj_wrap2(DTR, LTR,l)
        x0 = numpy.zeros(DTR.shape[0]+1)
        v,J,d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0,approx_grad=True)
        w = v[0:DTR.shape[0]]
        b = v[-1]
        STE = numpy.dot(w.T, DTE) + b
        STE = [(lambda i: 1 if i > 0 else 0)(i) for i in STE]
        num_correct_predicitions = correct_predictions(STE, LTE)

        acc = num_correct_predicitions/len(LTE)
        err = 1-acc
        print(l, J, end=' ')            #print J(w*,b*)
        print("%.1f%%" % (err*100))     #Error rate