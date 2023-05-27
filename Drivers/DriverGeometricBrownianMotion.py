from StandardWienerProcess import *
import matplotlib.pyplot as plt
from GeometricBrownianMotion import drift, diffusion, Jacobian

# Setup for Wiener Process
Tn = 10     
N = 30000
nW = 1
Ns = 5

T, W, dW = StandardWienerProcess(Tn,N,nW,Ns,seed=None)

# SDE solve using the EulerMaruyama ExplicitExplicit

p = [0.5,0.5]          # Parameter
x0 = [1]          # Initial State

#%%

from EulerMaruyamaExplicitExplicit import Solve

X = Solve(drift,diffusion,T,x0,W,args=(p,))

plt.figure(1)
for i in range(Ns):
    # We unpack each state
    X1 = X[:,0,i]
    plt.plot(T,X1)

plt.xlabel('Time')
plt.ylabel('Xt')
plt.title("Geometric BM: Euler-Maruyama Explicit Explicit")



#%%

from EulerMaruyamaImplicitExplicit import Solve

X = Solve(drift,diffusion,Jacobian,T,x0,W,args=(p,))

plt.figure(2)
for i in range(Ns):
    # We unpack each state
    X1 = X[:,0,i]
    plt.plot(T,X1)
    
plt.xlabel('Time')
plt.ylabel('Xt')
plt.title("Geometric BM: Euler-Maruyama Implicit Explicit")