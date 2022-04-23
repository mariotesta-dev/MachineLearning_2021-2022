import scipy.linalg
import numpy
import matplotlib
import matplotlib.pyplot as plt

def vcol(v): #v is a row vector(1, v.size), the function returns a column vector (v.size,1)
    return v.reshape((v.size, 1))

def vrow(v): #v is a column vector(v.size,1), the function returns a row vector (1,v.size)
    return v.reshape((1,v.size))

def compute_empirical_mean(X):
    return vcol(X.mean(1))

def compute_empirical_cov(D):
    mu = compute_empirical_mean(D)
    DC = D-mu
    C = numpy.dot(DC,DC.T) / D.shape[1]
    return C 

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
    
#computes log-likelihood which is the sum of all log-densities
def loglikelihood(X,mu,C):
    return logpdf_GAU_ND_Opt(X,mu,C).sum()

#computes vector of all densities of dataset X
def pdfND(X,mu,C):
    return numpy.exp(logpdf_GAU_ND_Opt(X,mu,C))
    
if __name__ == '__main__':
    
    #log density example for 1-D dataset XPlot
    XPlot = numpy.linspace(-8, 12, 1000)
    #assuming that mean = 1 and covariance = 2, must be 1x1 numpy arrays!
    m = numpy.ones((1,1)) * 1.0
    C = numpy.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND_Opt(vrow(XPlot), m, C)))
    plt.show() 


    #Maximum Likelihood Estimate
    #estimating mean and covariance as we did in the Lab3
    
    X = numpy.load("XND.npy") #loading sample data
    
    m_ML = compute_empirical_mean(X)
    C_ML = compute_empirical_cov(X)
    print(m_ML) 
    #[[-0.07187197] 
    # [ 0.05979594]]
    print(C_ML) 
    #[[0.94590166 0.09313534]
    # [0.09313534 0.8229693 ]]
    
    ll = loglikelihood(X, m_ML, C_ML) 
    #the higher this value the higher is the probability that estimated
    # mean and cov (m_ML, C_ML) are correct (so, close to real ones)
    print(ll)
    
    

