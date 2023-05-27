from StandardWienerProcess import *
import matplotlib.pyplot as plt
from VanDerPol import drift, diffusion,JacobianSDE

# Setup for Wiener Process
Tn = 20     
N = 30000
nW = 2
Ns = 2

T, W, dW = StandardWienerProcess(Tn,N,nW,Ns,seed=None)

# SDE solve using the EulerMaruyama ExplicitExplicit

p = [3,1]               # Parameter
x0 = [0.5,0.5]          # Initial State
tspan = [0, 20]

#%%

from EulerMaruyamaExplicitExplicit import Solve
T, X = Solve(drift,diffusion,tspan,x0,W,args=(p,))

fig, ax = plt.subplots(nrows=1,ncols=2)
fig.suptitle("VanderPol Model: Euler-Maruyama Explicit Explicit", fontsize=12)

for i in range(Ns):
# We unpack each state
    X1, X2 = X[:,0,i], X[:,1,i]
    ax[0].plot(T,X1)
    ax[1].plot(T,X2)


ax[0].set_xlabel('Time')
ax[0].set_ylabel('Counts')

ax[1].set_xlabel('Time')
ax[1].set_ylabel('Counts')

fig.tight_layout()
fig.subplots_adjust(top=0.92)


#%%

from EulerMaruyamaImplicitExplicit import Solve


T,X = Solve(drift,diffusion,JacobianSDE,tspan,x0,W,args=(p,))

fig, ax = plt.subplots(nrows=1,ncols=2)
fig.suptitle("VanderPol Model: Euler-Maruyama Implicit Explicit", fontsize=12)

for i in range(Ns):
# We unpack each state
    X1, X2 = X[:,0,i], X[:,1,i]
    ax[0].plot(T,X1)
    ax[1].plot(T,X2)


ax[0].set_xlabel('Time')
ax[0].set_ylabel('Counts')

ax[1].set_xlabel('Time')
ax[1].set_ylabel('Counts')

fig.tight_layout()
fig.subplots_adjust(top=0.92)

