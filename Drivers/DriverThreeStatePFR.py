import matplotlib.pyplot as plt
import numpy as np
from ThreeStatePFR import Model, Jacobian

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
v    = 1
D   = 0.1

# Parameter vector
p = np.array([k0,E,B,cAin,cBin,Tin,Nz,dz,v,D])     

# Set up initial conditions
cA0 = np.zeros([Nz,1])
cB0 = np.zeros([Nz,1])
T0  = np.zeros([Nz,1]) + 273.65

# Initial condition
x0 = np.row_stack([cA0,cB0,T0])[:,0]
x0 = x0.tolist()

# time interval                                       
tspan = [0,5]                                            

abstol = 10**(-2)          # Absolute Tolerance
reltol = 10**(-2)          # Relative Tolerance

#%%

from DormandPrince54 import Solve

T,X,E = Solve( Model, tspan, x0, 0.005, abstol, reltol , args=(p,))

cA   = X[:,:Nz] 
cB   = np.array(X[:,Nz:2*Nz])
Temp = np.array(X[:,2*Nz:])

#%%

from ESDIRK23 import Solve

T,X,E,H,R = Solve( Model, Jacobian, tspan, x0, 0.005, abstol, reltol , args=(p,))

cA   = X[:,:Nz] 
cB   = np.array(X[:,Nz:2*Nz])
Temp = np.array(X[:,2*Nz:])

#%%

x = np.linspace(0,L,Nz)

Tcoor,Xcoor = np.meshgrid(T,x)

plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(nrows=1,ncols=3)

cf = ax[0].contourf(Tcoor, Xcoor, cA.T, cmap="plasma")
plt.colorbar(cf, ax=ax[0])
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Pipe Position [m]")
ax[0].set_title("C_A [mol/L]")

cf = ax[1].contourf(Tcoor, Xcoor, cB.T, cmap="plasma")
plt.colorbar(cf, ax=ax[1])
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Pipe Position [m]")
ax[1].set_title("C_B [mol/L]")

cf = ax[2].contourf(Tcoor, Xcoor, Temp.T)
plt.colorbar(cf, ax=ax[2])
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("Pipe Position [m]")
ax[2].set_title("Temp. [Celsius]")

plt.subplots_adjust(wspace=0.5)
fig.suptitle(f'Three state PFR: Solution')
