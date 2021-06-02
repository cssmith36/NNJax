import numpy as np
import minresQLP2 as MQLP
import torch
import jax as jnp
#import RBM

#print(list(RBM.Test.parameters()))

a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.kron(a,a))

#a = a.flatten()
#print(a.reshape((2,3)))
#print(a.flatten())
#a = np.concatenate(([a],[b]), axis = 0)
#q = jnp.numpy.multiply(a,b)

#print(q)
#print(q[0])

'''https://github.com/pascanur/theano_optimize'''

a = np.array([1,1,1,-1])

A = np.outer(a,a)
print(A)
b = np.array([3,5,9,3])

AFun = lambda x: ([np.dot(A, x)])

print(type(AFun))

X = MQLP.MinresQLP(AFun,b,1e-6,100)
print(X)

inv = np.linalg.pinv(A)
print(inv)
print(np.matmul(inv,b))