import matplotlib.pyplot as plt
import numpy as np
from OneStatePFR import Model, Jacobian
from scipy.integrate import solve_ivp as solve

# Parameters
k0   = np.exp(24.6)
E    = 8500
B    = 560/4.186
cAin = 1.6/2
cBin = 2.4/2
Tin  = 273.65
Nz   = 50
L    = 10
dz   = L/Nz
v    = 100
D    = 0.1

# Parameter vector
p = np.array([k0,E,B,cAin,cBin,Tin,Nz,dz,v,D])     

# Set up initial conditions
T0  = np.zeros([Nz]) + 273.65

# Initial condition
x0 = T0

# time interval
tspan = [0,50]                                            

abstol = 10**(-8)          # Absolute Tolerance
reltol = 10**(-8)          # Relative Tolerance

J = Jacobian(0,T0,p)

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

cf = ax.contourf(Tcoor, Xcoor, X-273.15)
plt.colorbar(cf, ax=ax)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Pipe Position: z [m]")
ax.set_title("Temp. [Celsius]")

plt.subplots_adjust(wspace=0.5)
fig.suptitle(f'Three state PFR: Explicit RK45 Solution (v = {v} m/s)')

#%%

sol = solve(Model, tspan, x0, args=(p,), rtol = reltol, atol = abstol, method='Radau', jac=Jacobian)

T = sol.t               # The time points
X = sol.y               # The state vector


Nfev_Exp = sol.nfev
N_Exp = len(T)

x = np.linspace(0,L,Nz)

Tcoor,Xcoor = np.meshgrid(T,x)

plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(nrows=1,ncols=1)

cf = ax.contourf(Tcoor, Xcoor, X-273.15)
plt.colorbar(cf, ax=ax)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Pipe Position: z [m]")
ax.set_title("Temp. [Celsius]")

plt.subplots_adjust(wspace=0.5)
fig.suptitle(f'Three state PFR: Explicit RK45 Solution (v = {v} m/s)')
