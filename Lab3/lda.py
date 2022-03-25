import numpy
import sklearn.datasets
import matplotlib.pyplot as plt
import scipy.linalg

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

#Get dataset
def load_data():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def compute_within_covariance(D,L):
    SW = []
    for i in range(3):
        class_samples = D[:, L==i]
        C = compute_covariance(class_samples)
        SW.append(class_samples.shape[1]*C)
    return sum(SW) / float(D.shape[1])


#Computing covariance matrix
def compute_covariance(D):
    mu = D.mean(1)                    #find mean
    DC = D - vcol(mu)                 #centering data
    C = numpy.dot(DC, DC.T) / float(D.shape[1]) #covariance
    
    return C

def compute_between_covariance(D,L):
    SB = []
    mu = D.mean(1)
    for i in range(3):
        class_samples = D[:, L==i]
        mu_c = class_samples.mean(1)
        diff = vcol(mu_c) - vcol(mu)
        
        SB.append(class_samples.shape[1]*diff*diff.T)
    return sum(SB) / float(D.shape[1])

def compute_LDA_direction(SW, SB, m):
    U, s, _ = numpy.linalg.svd(SW)
    P1 = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)
    SBT = numpy.dot(P1, numpy.dot(SB, P1.T))
    U2, s2, _ = numpy.linalg.svd(SBT)
    P2 = U2[:, 0:m]
    return numpy.dot(P1.T,P2)

def generalized_LDA_direction(SW, SB, m):
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    return W


def plot_data(DP, L):
    dp0 = DP[:, L == 0]
    dp1 = DP[:, L == 1]
    dp2 = DP[:, L == 2]

    plt.figure()
    plt.scatter(dp0[0], dp0[1], label = 'Setosa')
    plt.scatter(dp1[0], dp1[1], label = 'Versicolor')
    plt.scatter(dp2[0], dp2[1], label = 'Virginica')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    D,L = load_data()
    SW = compute_within_covariance(D,L)
    SB = compute_between_covariance(D,L)
    W_check = numpy.load("IRIS_LDA_matrix_m2.npy")
    W = generalized_LDA_direction(SW, SB, 2)
    #this works, but the result is inverted in the axes
    #W = compute_LDA_direction(SW,SB,2)
    DP = numpy.dot(W.T, D) #apply projection
    plot_data(DP,L)








    




