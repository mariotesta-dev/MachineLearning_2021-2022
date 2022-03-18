import numpy
import sklearn.datasets
import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size, 1))

#Get dataset
def load_data():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

#Computing covariance matrix
def compute_covariance(D):
    mu = D.mean(1)                    #find mean
    DC = D - vcol(mu)                 #centering data
    C = numpy.dot(DC, DC.T) / float(D.shape[1]) #covariance
    
    return C

def plot_data(DP, L):
    dp0 = DP[:, L == 0]
    dp1 = DP[:, L == 1]
    dp2 = DP[:, L == 2]

    plt.figure()
    plt.scatter(dp0[0], -dp0[1], label = 'Setosa')
    plt.scatter(dp1[0], -dp1[1], label = 'Versicolor')
    plt.scatter(dp2[0], -dp2[1], label = 'Virginica')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    D,L = load_data()
    C = compute_covariance(D)

    #compute eigenvectors and eigenvalues using numpy function
    #which returns the eigenvalues (s), sorted from smallest to largest,
    # and the corresponding eigenvectors (columns of U)
    #(s, U) = numpy.linalg.eigh(C)
    U, s, Vh = numpy.linalg.svd(C)
    U_check = numpy.load("IRIS_PCA_matrix_m4.npy")
    m = 4
    P = U[:, 0:m]

    DP = numpy.dot(P.T, D) #apply projection
    plot_data(DP,L)






    




