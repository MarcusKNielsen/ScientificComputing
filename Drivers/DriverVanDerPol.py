import matplotlib.pyplot as plt
import numpy as np
from VanDerPol import Model, Jacobian

p = 10              # Parameter
x0 = [2,0]          # Initial State
tspan = [0, 20]     # Time Interval

abstol = 10**(-3)          # Absolute Tolerance
reltol = 10**(-3)          # Relative Tolerance


#%% ExplicitEulerFixedStepSize

from ExplicitEulerFixedStepSize import Solve

N = 100

T,X = Solve(Model,tspan,x0,N,args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

# Plot of the Solution
plt.figure(1)
plt.plot(T,X1)
plt.plot(T,X2)
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("VanderPol Model: Explicit Solve")


#%% ImplicitEulerFixedStepSize

from ImplicitEulerFixedStepSize import Solve

N = 10000

T,X = Solve(Model,Jacobian,tspan,x0,N,args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

# Plot of the Solution
plt.figure(2)
plt.plot(T,X1)
plt.plot(T,X2)
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("VanderPol Model: Implicit Solve")


#%%

from ExplicitEulerAdaptiveStepSize import Solve

T,X = Solve( Model, tspan, x0, 0.005, abstol, reltol , args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(3)
plt.plot(T,X1)
plt.plot(T,X2)
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("VanderPol Model: Explicit Adaptive Step Size Solve")


#%%

from ImplicitEulerAdaptiveStepSize import Solve

T,X = Solve( Model, Jacobian, tspan, x0, 0.05, abstol, reltol , args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(4)
plt.plot(T,X1)
plt.plot(T,X2)
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("VanderPol Model: Implicit Adaptive Step Size Solve")


#%%

from RungeKutta4AdaptiveStepSize import Solve

T,X = Solve( Model, tspan, x0, 0.005, abstol, reltol , args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(5)
plt.plot(T,X1)
plt.plot(T,X2)
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("VanderPol Model: RK4 Adaptive Step Size Solve")

#%%

from ESDIRK23 import Solve

T,X,E,H,R = Solve( Model, Jacobian, tspan, x0, 0.05, abstol, reltol , args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(4)
plt.plot(T,X1)
plt.plot(T,X2)
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("VanderPol Model: ESDIRK23 Solve")

#%%




