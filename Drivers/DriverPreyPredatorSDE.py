from StandardWienerProcess import *
import matplotlib.pyplot as plt
from PreyPredator import Drift, Diffusion, JacobianDrift

# Setup for Wiener Process
Tn = 20      
N = 30000
nW = 2
Ns = 10

T, W, dW = StandardWienerProcess(Tn,N,nW,Ns,seed=None)



p     = [1,1,1,1,0.1,0.1]   # Parameter
x0    = [2,1]               # Initial State
tspan = [0, 50]             # Time spand


#%%

# SDE solve using the EulerMaruyama ExplicitExplicit
from EulerMaruyamaExplicitExplicit import Solve
T, X = Solve( Drift, Diffusion, tspan, x0, W, args=(p,))


fig, ax = plt.subplots(nrows=1,ncols=2)
fig.suptitle("Prey Predator: Euler-Maruyama Explicit Explicit", fontsize=12)

for i in range(Ns):
# We unpack each state
    X1, X2 = X[:,0,i], X[:,1,i]
    ax[0].plot(T,X1)
    ax[1].plot(T,X2)


ax[0].set_xlabel('Time')
ax[0].set_ylabel('Counts')
ax[0].set_title('Prey')

ax[1].set_xlabel('Time')
ax[1].set_ylabel('Counts')
ax[1].set_title('Predator')

fig.tight_layout()
fig.subplots_adjust(top=0.85)


#%%


# SDE solve using the EulerMaruyama ImplicitExplicit
from EulerMaruyamaImplicitExplicit import Solve
T, X = Solve( Drift, Diffusion, JacobianDrift, tspan, x0, W, args=(p,))


fig, ax = plt.subplots(nrows=1,ncols=2)
fig.suptitle("Prey Predator: Euler-Maruyama Implicit Explicit", fontsize=12)

for i in range(Ns):
# We unpack each state
    X1, X2 = X[:,0,i], X[:,1,i]
    ax[0].plot(T,X1)
    ax[1].plot(T,X2)


ax[0].set_xlabel('Time')
ax[0].set_ylabel('Counts')
ax[0].set_title('Prey')

ax[1].set_xlabel('Time')
ax[1].set_ylabel('Counts')
ax[1].set_title('Predator')

fig.tight_layout()
fig.subplots_adjust(top=0.85)


