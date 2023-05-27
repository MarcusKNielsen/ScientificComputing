import matplotlib.pyplot as plt
import numpy as np
from PreyPredator import Model, Jacobian


p = np.array([1,1,1,1])     # Parameters
x0 = np.array([1,2])        # Initial condition
tspan = [0,10]              # time interval

abstol = 10**(-8)          # Absolute Tolerance
reltol = 10**(-8)          # Relative Tolerance

#%% Explicit solve using RK45 scipy solve function

from scipy.integrate import solve_ivp as solve

sol = solve(Model, tspan, x0, args=(p,), rtol = reltol, atol = abstol)

T = sol.t                   # The time points
X = sol.y                   # The state vector

Nfev   = sol.nfev
Nsteps = len(T)


#%% Implicit solve using scipy solve function

#from scipy.integrate import solve_ivp as solve

#sol = solve(Model, tspan, x0, args=(p,), rtol = reltol, atol = abstol, method="Radau", jac=Jacobian)

#T = sol.t                   # The time points
#X = sol.y                   # The state vector

#%% ExplicitEulerFixedStepSize

from ExplicitEulerFixedStepSize import Solve

N = 10000

T,X,Nfev = Solve(Model,tspan,x0,N,args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

# Plot of the Solution
plt.figure(1)
plt.plot(T,X1,label="Prey")
plt.plot(T,X2,label="Predators")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("Prey-Predator Model: Explicit Solve")
plt.legend()

#%% ImplicitEulerFixedStepSize

from ImplicitEulerFixedStepSize import Solve

N = 10000

T,X = Solve(Model,Jacobian,tspan,x0,N,args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

# Plot of the Solution
plt.figure(2)
plt.plot(T,X1,label="Prey")
plt.plot(T,X2,label="Predators")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("Prey-Predator Model: Implicit Solve")
plt.legend()

#%%

from ExplicitEulerAdaptiveStepSize import Solve

T,X,Nfev,R,H,Naccept,Nreject = Solve( Model, tspan, x0, 0.05, abstol, reltol , args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(3)
plt.plot(T,X1,label="Prey")
plt.plot(T,X2,label="Predators")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("Prey-Predator Model: Explicit Adaptive Step Size Solve")
plt.legend()

#%%
from test123 import Solve
import time
start_time = time.time()
T, X, Nfev, Njac, Nlu, R, H, Naccept, Nreject = Solve( Model, Jacobian, tspan, x0, 0.05, abstol, reltol , args=(p,))
print("--- %s seconds ---" % (time.time() - start_time))
# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(3)
plt.plot(T,X1,label="Prey")
plt.plot(T,X2,label="Predators")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("Prey-Predator Model: Explicit Adaptive Step Size Solve")
plt.legend()



#%%

from ImplicitEulerAdaptiveStepSize import Solve
import time
start_time = time.time()
T,X,Nfev,R,H,Naccept,Nreject = Solve( Model, Jacobian, tspan, x0, 0.005, abstol, reltol , args=(p,))
print("--- %s seconds ---" % (time.time() - start_time))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(4)
plt.plot(T,X1,label="Prey")
plt.plot(T,X2,label="Predators")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("Prey-Predator Model: Explicit Adaptive Step Size Solve")
plt.legend()

#%%

from ClassicRungeKutta import Solve

N = 10000

T,X = Solve(Model,tspan,x0,N,args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(3)
plt.plot(T,X1,label="Prey")
plt.plot(T,X2,label="Predators")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("Prey-Predator Model: Explicit Adaptive Step Size Solve")
plt.legend()

#%%

from RungeKutta4AdaptiveStepSize import Solve

T,X = Solve( Model, tspan, x0, 0.005, abstol, reltol , args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(5)
plt.plot(T,X1,label="Prey")
plt.plot(T,X2,label="Predators")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("Prey-Predator Model: RK4 Adaptive Step Size Solve")
plt.legend()

#%%

from DormandPrince54 import Solve

T,X,E = Solve( Model, tspan, x0, 0.005, abstol, reltol , args=(p,))

# We unpack each state
X1, X2 = X[:,0], X[:,1]

plt.figure(5)
plt.plot(T,X1,label="Prey")
plt.plot(T,X2,label="Predators")
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title("Prey-Predator Model: DP54 Adaptive Step Size Solve")
plt.legend()

plt.figure(6)
plt.plot(T,E)

