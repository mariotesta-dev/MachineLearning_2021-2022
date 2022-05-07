import numpy
import scipy.optimize

def fun(x):
    y,z = x[0], x[1]
    expr = (y+3)**2 + numpy.sin(y) + (z+1)**2
    return expr

def fun_grad(x):
    y,z = x[0], x[1]
    expr = (y+3)**2 + numpy.sin(y) + (z+1)**2
    grad = numpy.array([2*(y+3)+numpy.cos(y),2*(z+1)])
    return expr,grad

if __name__ == '__main__':

    x0 = numpy.array([0,0])          #starting value of algorithm
    x,f,d = scipy.optimize.fmin_l_bfgs_b(fun,x0, approx_grad=True)
    x1,f1,d1 = scipy.optimize.fmin_l_bfgs_b(fun_grad,x0)

    print("approx_grad=True :")
    print(x,f)
    print("computing fun_grad :")
    print(x1,f1)
