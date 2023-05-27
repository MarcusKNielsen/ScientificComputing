import numpy as np
from numpy import exp

def Model(t,x,p):
    
    # Unpack parameter vector
    k0,E,B,cAin,cBin,Tin,Nz,dz,v,D= p
    
    Nz = int(Nz)
    
    # Unpack states
    T  = x
    '''
    ### Implementation of heat T
    '''
    # Advection of heat T
    NadvT     = np.zeros([Nz+1])
    NadvT[0]  = v*Tin
    NadvT[1:] = v*T[:]
    
    # Diffusion of heat T
    JT       = np.zeros([Nz+1])
    JT[1:Nz] = (-D/dz) * (T[1:] - T[:Nz-1])
    
    # Flux = advection + diffusion for heat T
    NT = NadvT + JT

    # Concentrations
    cA = cAin + (1/B)*(Tin - T)
    cB = cBin + (2/B)*(Tin - T)
    

    # Production rate of T
    RT  = B*k0 * exp(-E/T) * cA * cB
     
    # Advection-Diffusion-Reaction Equation for heat T
    Tdot = (NT[1:] - NT[:Nz])/(-dz) + RT

    
    return Tdot
    

def Jacobian(t,x,p):
    # Unpack parameter vector
    k0,E,B,cAin,cBin,Tin,Nz,dz,v,D = p
    
    Nz = int(Nz)
    
    # Unpack states
    T  = x
    
    # Concentrations
    cA = cAin + (1/B)*(Tin - T)
    cB = cBin + (2/B)*(Tin - T)
    
    # Initialize Jacobian
    Jac = np.zeros([Nz,Nz])
    
    for j in range(Nz): # Row
        for i in range(Nz): # Column

            # dT/dT
            if j == 0:
                if i == 0:
                    Jac[j,i] = (-1/dz)*(v + D/dz) + B*k0*(-E/T[0])*exp(-E/T[0])*cA[0]*cB[0]
                elif i == 1:
                    Jac[j,i] = D/dz**2
            elif 0 < j < (Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) + B*k0*(-E/T[j])*exp(-E/T[j])*cA[i]*cB[j]
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
            elif j == (Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) + B*k0*(-E/T[Nz-1])*exp(-E/T[Nz-1])*cA[Nz-1]*cB[Nz-1]
                
    return Jac


def ModelJacobian(t,x,p):
    
    # Unpack parameter vector
    k0,E,B,cAin,cBin,Tin,Nz,dz,v,D= p
    
    Nz = int(Nz)
    
    # Unpack states
    T  = x
    '''
    ### Implementation of heat T
    '''
    # Advection of heat T
    NadvT     = np.zeros([Nz+1])
    NadvT[0]  = v*Tin
    NadvT[1:] = v*T[:]
    
    # Diffusion of heat T
    JT       = np.zeros([Nz+1])
    JT[1:Nz] = (-D/dz) * (T[1:] - T[:Nz-1])
    
    # Flux = advection + diffusion for heat T
    NT = NadvT + JT

    # Concentrations
    cA = cAin + (1/B)*(Tin - T)
    cB = cBin + (2/B)*(Tin - T)
    

    # Production rate of T
    RT  = B*k0 * exp(-E/T) * cA * cB
     
    # Advection-Diffusion-Reaction Equation for heat T
    Tdot = (NT[1:] - NT[:Nz])/(-dz) + RT
    
    # Initialize Jacobian
    Jac = np.zeros([Nz,Nz])
    
    for j in range(Nz): # Row
        for i in range(Nz): # Column

            # dT/dT
            if j == 0:
                if i == 0:
                    Jac[j,i] = (-1/dz)*(v + D/dz) + B*k0*(-E/T[0])*exp(-E/T[0])*cA[0]*cB[0]
                elif i == 1:
                    Jac[j,i] = D/dz**2
            elif 0 < j < (Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) + B*k0*(-E/T[j])*exp(-E/T[j])*cA[i]*cB[j]
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
            elif j == (Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) + B*k0*(-E/T[Nz-1])*exp(-E/T[Nz-1])*cA[Nz-1]*cB[Nz-1]
                    
    return Tdot, Jac