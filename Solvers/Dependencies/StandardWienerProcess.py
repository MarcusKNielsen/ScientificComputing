import numpy as np


def StandardWienerProcess(tspan,N,nW,Ns,seed=None):
    
    # tspan time interval
    # N is the number of points T will be split up in
    # nW is the dimension of the brownian motion
    # Ns is the number of simulations
    
    dt = (tspan[1] - tspan[0])/N
    
    if isinstance(seed, (int, float)):
        np.random.seed(seed)
    
    dW = np.random.normal(loc=0, scale=1, size=[N+1,nW,Ns])
    T = np.zeros(N+1)
    W = np.zeros([N+1,nW,Ns])
    for i in range(N):
        W[i+1,:,:] = W[i,:,:] + np.sqrt(dt)*dW[i,:,:]
        T[i+1] = T[i] + dt
    
    return T, W, dW
    

