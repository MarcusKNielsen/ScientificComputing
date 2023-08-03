import numpy as np
from numpy import exp

def Model(t,x,p):
    
    # Unpack parameter vector
    Nz,dz,theta,mu,D = p
    
    Nz = int(Nz)
    
    z = (np.linspace(1,Nz,Nz) - 0.5)*dz
    v = theta*(mu-z)
    
    # Unpack states
    T = x
    '''
    ### Implementation of heat T
    '''
    # Advection of heat T
    NadvT       = np.zeros([Nz+1])
    NadvT[0]    = 0
    NadvT[1:]   = v*T[:]
    NadvT[-1]   = 0
    
    # Diffusion of heat T
    JT       = np.zeros([Nz+1])
    JT[1:Nz] = (-D/dz) * (T[1:] - T[:Nz-1])
    
    # Flux = advection + diffusion for heat T
    NT = NadvT + JT
     
    # Advection-Diffusion Equation for heat T
    Tdot = (NT[1:] - NT[:Nz])/(-dz)

    
    return Tdot
    

def Jacobian(t,x,p):
    
    # Unpack parameter vector
    Nz,dz,theta,mu,D = p
    
    Nz = int(Nz)
    
    z = (np.linspace(1,Nz,Nz) - 0.5)*dz
    v = theta*(mu-z)
    
    
    # Initialize Jacobian
    Jac = np.zeros([Nz,Nz])
    
    for j in range(Nz): # Row
        for i in range(Nz): # Column

            # dT/dT
            if j == 0:
                if i == 0:
                    Jac[j,i] = (-1/dz)*(v[j] + D/dz)
                elif i == 1:
                    Jac[j,i] = D/dz**2
            elif 0 < j < (Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v[j] + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v[j] + 2*D/dz)
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
            elif j == (Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v[j] + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v[j] + 2*D/dz)
                
    return Jac