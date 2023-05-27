import matplotlib.pyplot as plt
import numpy as np
from OneStateTank import Model, Jacobian

# Parameters
F    = 0.0001
V    = 0.105
k0   = np.exp(24.6)
E    = 8500
B    = 560/4.186
cAin = 1.6/2
cBin = 2.4/2
Tin  = 273.65

p = np.array([F,V,k0,E,B,cAin,cBin,Tin])     # Parameters
x0 = [1,2,3]                                 # Initial condition
tspan = [0,10]                               # time interval

abstol = 10**(-8)          # Absolute Tolerance
reltol = 10**(-8)          # Relative Tolerance

#%%
from ImplicitEulerAdaptiveStepSize import Solve

T,X = Solve( Model, Jacobian, tspan, x0, 0.005, abstol, reltol , args=(p,))

# We unpack each state
X1 = X

plt.figure(5)
plt.plot(T,X1,label="T")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("OneStateTank Model: DP54 Adaptive Step Size Solve")
plt.legend()



#%%
from DormandPrince54 import Solve

T,X = Solve( Model, tspan, x0, 0.005, abstol, reltol , args=(p,))

# We unpack each state
X1 = X

plt.figure(5)
plt.plot(T,X1,label="T")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("OneStateTank Model: DP54 Adaptive Step Size Solve")
plt.legend()
