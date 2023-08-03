import matplotlib.pyplot as plt
import numpy as np
from AdvectionDiffusion import Model, Jacobian

# Parameters
Nz    = 150
L     = 10
dz    = L/Nz
mu    = 7
D     = 0.1
theta = 1

# Parameter vector
p = np.array([Nz,dz,theta,mu,D])     

# Set up initial conditions
T0 = np.zeros([Nz])
T0[25:50] = 1

# Initial condition
x0 = T0

# time interval
tspan = [0,50]                                            

abstol = 10**(-6)          # Absolute Tolerance
reltol = 10**(-6)          # Relative Tolerance

h0 = 0.005


#%%

from ESDIRK23 import Solve

T,X,Nfev,Njac,Nlu,R,H,Naccept,Nreject = Solve(Model,Jacobian, tspan, x0, h0, abstol, reltol , args=(p,))

X = X.T

#%%

threshold = 0.00001
X[X < threshold] = 0
#%%

print(f"Nsteps  = {len(T)-1}")
print(f"Nfev    = {Nfev}")
print(f"Njac    = {Njac}")
print(f"Nlu     = {Nlu}")
print(f"Naccept = {Naccept}")
print(f"Nreject = {Nreject}")

x = np.linspace(0,L,Nz)

Tcoor,Xcoor = np.meshgrid(T,x)

cf = plt.contourf(Tcoor, Xcoor, X)
plt.colorbar(cf)
plt.xlabel("Time [s]")
plt.ylabel("Pipe Position: z [m]")
plt.title("Temp. [Celsius]")


#%%

plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].plot(T,H,label="step size",linewidth=2.5)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('h')
ax[0].set_title("Change in step size")
ax[0].legend(loc='upper left')

ax[1].plot(T,R,label="r",linewidth=2.5)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('r - value')
ax[1].set_title("Error estimator")
ax[1].legend(loc='lower right')

fig.suptitle('Adaptive Step Size Controller')
