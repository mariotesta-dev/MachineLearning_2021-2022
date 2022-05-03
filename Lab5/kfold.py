import numpy
import sklearn.datasets
#import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

#Get IRIS dataset
def load_data():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def kfold(D, L, k):
    data_folds = numpy.array_split(D, k, axis=1)
    label_folds = numpy.array_split(L, k, axis=0)

    d_train = []
    d_test = []

    cross_val_data = {'train': d_train, 'test':d_test}
    for i, test_i in enumerate(data_folds):
        d_train.append(data_folds[:i] + data_folds[i+1:])
        d_test.append(test_i)
    
    l_train = []
    l_test = []

    cross_val_label = {'train': l_train, 'test':l_test}
    for i, test_i in enumerate(label_folds):
        l_train.append(label_folds[:i] + label_folds[i+1:])
        l_test.append(test_i)
    return cross_val_data, cross_val_label


def compute_empirical_mean(X):
    return vcol(X.mean(1))

#the Naive Bayes version of the MVG is simply a Gaussian classifer
#where the covariance matrices are diagonal, since the number of features is small
#we can just multiply C with the identity matrix
def compute_diag_empirical_cov(D):
    mu = compute_empirical_mean(D)
    DC = D-mu
    C = numpy.dot(DC,DC.T) / D.shape[1]
    return C * numpy.eye(C.shape[0], C.shape[1])

def compute_empirical_cov(D):
    mu = compute_empirical_mean(D)
    DC = D-mu
    C = numpy.dot(DC,DC.T) / D.shape[1]
    return C 

#Computing covariance matrix
def compute_covariance(D):
    mu = D.mean(1)                    #find mean
    DC = D - vcol(mu)                 #centering data
    C = numpy.dot(DC, DC.T) / float(D.shape[1]) #covariance
    
    return C

#In the Tied versin, the ML solution for the covariance matrix is given by the
# empirical within-class covariance matrix
def compute_within_covariance_tied_naive(D,L):
    SW = []
    for i in range(3):
        class_samples = D[:, L==i]
        C = compute_covariance(class_samples)
        SW.append(class_samples.shape[1]*C)
    cov = sum(SW) / float(D.shape[1])
    return cov * numpy.eye(cov.shape[0], cov.shape[1])

#In the Tied versin, the ML solution for the covariance matrix is given by the
# empirical within-class covariance matrix
def compute_within_covariance(D,L):
    SW = []
    for i in range(3):
        class_samples = D[:, L==i]
        C = compute_covariance(class_samples)
        SW.append(class_samples.shape[1]*C)
    return sum(SW) / float(D.shape[1])

#computes estimates of mean and cov for each class, this is done with a generic approach
#instead of computing v0,v1,v2 using [:, LTR == 0...1...2] as done in the previous labs
def compute_classifier_params(DTR, LTR, alg):

    samples = []
    for label in list(set(LTR)):
        v = DTR[:, LTR == label]
        samples.append(v)
    
    params = []
    for v in samples:
        mu_ML = compute_empirical_mean(v)
        if(alg == 'naive'):
            C_ML = compute_diag_empirical_cov(v)
        elif(alg == 'mult'):
            C_ML = compute_empirical_cov(v)
        elif(alg == 'tied-cov'):
            C_ML = compute_within_covariance(DTR,LTR)
        elif(alg == 'tied-naive'):
            C_ML = compute_within_covariance_tied_naive(DTR,LTR)
        params.append([mu_ML,C_ML])
    
    return params

#logpdf_GAU_ND_Opt = takes X ([M x N] dataset) as an argument,
#then computes the vector of log densities for feature vector x (so, for each column of X).
def logpdf_GAU_ND_Opt(X,mu,C):
    C_inv = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]
    #the logarithmic determinant of C (covariance of X) can be computed using the slodget function:
    #it returns [0]= sign of determinand, [1]=abs value of logarithmic determinant
    
    Y = [] #vector of log-densities for each column of X: Y = [log N (x_1|µ, Σ),. . . ,log N (x_n |µ, Σ)]
    for i in range(X.shape[1]):
        x = X[:, i:i+1] #takes one column at a time
        res = const + -0.5 * numpy.dot( (x-mu).T, numpy.dot(C_inv, (x-mu)) )
        Y.append(res)
    return numpy.array(Y).ravel() #flatten and returns a 1-D array

def class_loglikelihood(DTE,classifier_params):
    S = []
    for params in classifier_params:
        class_loglike = pdfND(DTE, params[0], params[1])
        S.append(class_loglike)
    return numpy.array(S)

#computes vector of all densities of dataset X
def pdfND(X,mu,C):
    return numpy.exp(logpdf_GAU_ND_Opt(X,mu,C))

def correct_predictions(predicted_labels, LTE):
    correct = 0
    for i,x in enumerate(predicted_labels):
        if x == LTE[i]:
            correct += 1
    return correct


if __name__ == '__main__':
    D, L = load_data()
    #DTR and LTR are training data and labels, 
    #DTE and LTE are evaluation data and labels
    #alg can be: 'mult', 'naive', 'tied-cov', 'tied-naive'
    alg = 'tied-naive'
    k = 150
    data, label = kfold(D,L,k) #split D and L in k parts, sequence them in k iterations of k-1 training parts + 1 test part

    err_tot = 0.0

    for i in range(k):
        DTR = numpy.hstack(data['train'][i])
        LTR = numpy.concatenate(label['train'][i])
        DTE = data['test'][i]
        LTE = label['test'][i]
        classifier_params = compute_classifier_params(DTR,LTR,alg)
        S = class_loglikelihood(DTE,classifier_params)
        class_prior_probability = 1/3
        SJoint = S * class_prior_probability
        SMarginal = vrow(SJoint.sum(0))
        SPost = SJoint/SMarginal
        predicted_labels = numpy.argmax(SPost, axis=0)

        num_correct_predicitions = correct_predictions(predicted_labels, LTE)

        acc = num_correct_predicitions/len(LTE)
        err = 1-acc
        err_tot += err

    err_tot = err_tot/k
    print("-- %s --" % alg)
    print("error rate: %.1f%%" % (err_tot*100))
