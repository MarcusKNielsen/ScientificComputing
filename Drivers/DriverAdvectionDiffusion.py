import matplotlib.pyplot as plt
import numpy as np
from AdvectionDiffusion import Model, Jacobian
from scipy.integrate import solve_ivp as solve

# Parameters
Nz    = 700
L     = 10
dz    = L/Nz
mu    = 7
D     = 0.1
theta = 1

# Parameter vector
p = np.array([Nz,dz,theta,mu,D])     

# Set up initial conditions
T0 = np.zeros([Nz])
T0[75:125] = 1

# Initial condition
x0 = T0

# time interval
tspan = [0,20]                                            

abstol = 10**(-6)          # Absolute Tolerance
reltol = 10**(-6)          # Relative Tolerance

#J = Jacobian(0,T0,p)

#%%

sol = solve(Model, tspan, x0, args=(p,), rtol = reltol, atol = abstol, method='RK45')

T = sol.t               # The time points
X = sol.y               # The state vector


Nfev_Exp = sol.nfev
N_Exp = len(T)

x = np.linspace(0,L,Nz)

Tcoor,Xcoor = np.meshgrid(T,x)

plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(nrows=1,ncols=1)

cf = ax.contourf(Tcoor, Xcoor, X)
plt.colorbar(cf, ax=ax)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Pipe Position: z [m]")
ax.set_title("Temp. [Celsius]")

plt.subplots_adjust(wspace=0.5)
fig.suptitle(f'Three state PFR: Explicit RK45 Solution')