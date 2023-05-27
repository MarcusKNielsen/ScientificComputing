import matplotlib.pyplot as plt
import numpy as np
from SupplyDemandDynamics import Model

# Parameter values:
a = 10**(-3)    # The rate at which none interested people becomes interested in buying the assert
b = 100         # The rate at which people holding the assert want to sell it
c = 10**(-3)    # This is the rate a which trades happens
k = 0.01         # The rate at which the assert is lost
A = 10**(-7)    # The growth rate of the population
Q = 10**6       # The carrying capacity of the population
M = 1000      # The upper bound for the price of the assert

# The initial states:
x0 = 0          # The initial number of buyers
y0 = 0          # The initial number of sellers
v0 = 10**5      # The initial number of people not owning the assert and not interested in buying
z0 = 50         # The initial number of people owning the assert but not interested in selling
s0 = 1/100      # The initial price of the assert

p = np.array([a,b,c,k,A,Q,M]) # Parameters
x_init = [x0,y0,v0,z0,s0]     # Initial condition
tspan = [0,50]                # time interval

abstol = 10**(-6)          # Absolute Tolerance
reltol = 10**(-6)          # Relative Tolerance

from DormandPrince54 import Solve

T,X = Solve( Model, tspan, x_init, 0.01, abstol, reltol , args=(p,))

# We unpack each state
x, y, v, z, s = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]


#%%
plt.figure(5)
plt.plot(T,x,label="X: Buyers")
plt.plot(T,y,label="Y: Sellers")
plt.plot(T,v/10**4,label="V: Not interested")
plt.plot(T,z,label="Z: Holding the assert")
plt.plot(T,s*100,label="S: Price")
plt.xlabel('Time in years')
plt.ylabel('Counts')
plt.title("Supply-Demand: DP54 Adaptive Step Size Solve")
plt.legend()

