from StandardWienerProcess import *
import matplotlib.pyplot as plt
from Rating import drift, diffusion, Model
#%%
# Setup for Wiener Process
Tn = 100      
N = 10000
nW = 1
Ns = 5

T, W, dW = StandardWienerProcess(Tn,N,nW,Ns,seed=None)


p = [1,0.05]   # Parameter
x0 = [0.1]        # Initial State


#%%
from Rating import drift, diffusion, Model
from DormandPrince54 import Solve

p = [1,0.5,1,0]
x0 = 0.2
h0 = 0.1
abstol = 10**(-6)
reltol = 10**(-6)
tspan = [0, 10]

T,X = Solve(Model, tspan, x0, h0, abstol, reltol, args=(p,))

plt.plot(T,X)
#plt.axhline(0)
#plt.axhline(p)


#%%

# SDE solve using the EulerMaruyama ExplicitExplicit
from EulerMaruyamaExplicitExplicit import Solve
X = Solve(drift,diffusion,T,x0,W,args=(p,))


plt.figure(1)

for i in range(Ns):
# We unpack each state
    plt.plot(T,X[:,0,i])

#%%

x = np.linspace(0,10,1000)

plt.plot(x,diffusion(x, x, p))

#%%























