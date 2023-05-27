import numpy as np
from numpy import exp

def Model(t,x,p):
    
    # Unpack parameter vector
    k0,E,B,cAin,cBin,Tin,Nz,dz,v,D= p
    
    Nz = int(Nz)
    
    
    # Unpack states
    cA = np.array(x[:Nz])
    cB = np.array(x[Nz:2*Nz])
    T  = np.array(x[2*Nz:])
    

    '''
    ### Implementation of compound A
    '''
    # Advection of compound A
    NadvA     = np.zeros([Nz+1])
    NadvA[0]  = v*cAin
    NadvA[1:] = v*cA[:]
    
    # Diffusion of compound A
    JA       = np.zeros([Nz+1])
    JA[1:Nz] = (-D/dz) *  (cA[1:] - cA[:Nz-1])
    
    # Flux = advection + diffusion for compound A
    NA = NadvA + JA

    # Production rate of A
    r  = k0*exp(-E/T)*cA*cB
    RA = -r
    
    # Advection-Diffusion-Reaction Equation for Compound A
    cAdot = (NA[1:] - NA[:Nz])/(-dz) + RA
    
    '''
    ### Implementation of compound B
    '''
    # Advection of compound B
    NadvB     = np.zeros([Nz+1])
    NadvB[0]  = v*cBin
    NadvB[1:] = v*cB[:]
    
    # Diffusion of compound B
    JB       = np.zeros([Nz+1])
    JB[1:Nz] = (-D/dz) *  (cB[1:] - cB[:Nz-1])
    
    # Flux = advection + diffusion for compound B
    NB = NadvB + JB

    # Production rate of B
    r  = 2*k0*exp(-E/T)*cA*cB
    RB = -r
    
    # Advection-Diffusion-Reaction Equation for Compound B
    cBdot = (NB[1:] - NB[:Nz])/(-dz) + RB
    
    '''
    ### Implementation of heat T
    '''
    # Advection of heat T
    NadvT     = np.zeros([Nz+1])
    NadvT[0]  = v*Tin
    NadvT[1:] = v*T[:]
    
    # Diffusion of heat T
    JT       = np.zeros([Nz+1])
    JT[1:Nz] = (-D/dz) *  (T[1:] - T[:Nz-1])
    
    # Flux = advection + diffusion for heat T
    NT = NadvT + JT

    # Production rate of T
    r  = B*k0 * exp(-E/T) * cA * cB
    RT = r
    
    # Advection-Diffusion-Reaction Equation for heat T
    Tdot = (NT[1:] - NT[:Nz])/(-dz) + RT
    
    '''
    ### Formatting output
    '''
    cAdot = np.array([cAdot]).T
    cBdot = np.array([cBdot]).T
    Tdot  = np.array([ Tdot]).T
    
    xdot = np.row_stack([cAdot, cBdot, Tdot])
    xdot = xdot[:,0]
    
    return xdot
    

def Jacobian(t,x,p):
    
    # Unpack parameter vector
    k0,E,B,cAin,cBin,Tin,Nz,dz,v,D = p
    
    Nz = int(Nz)
    
    # Unpack states
    cA = np.array(x[:Nz])
    cB = np.array(x[Nz:2*Nz])
    T  = np.array(x[2*Nz:])
    
    # Initialize Jacobian
    N = 3*Nz
    Jac = np.zeros([N,N])
    
    for j in range(N): # Row
        for i in range(N): # Column
        
            # dcA/dcA
            if j == 0:
                if i == 0:
                    Jac[j,i] = (-1/dz)*(v + D/dz) - k0*exp(-E/T[i])*cB[i]
                elif i == 1:
                    Jac[j,i] = D/dz**2
                    
            elif 0 < j <(Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) - k0*exp(-E/T[j])*cB[j]
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
                    
            elif j == (Nz-1):
                if i == (Nz-2):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == (Nz-1):
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) - k0*exp(-E/T[j])*cB[j]

            # dcA/dcB
            if 0<=j<Nz and i==(Nz+j):
                Jac[j,i] = - k0*exp(-E/T[j])*cA[j]

            # dcA/dT
            if 0<=j<Nz and i==(2*Nz+j):
                Jac[j,i] = - k0*(-E/T[j])*exp(-E/T[j])*cA[j]*cB[j]
                
            # dcB/dcA
            if Nz<=j<(2*Nz) and i==(j-Nz):
                Jac[j,i] = -2*k0*exp(-E/T[i])*cB[i]
                
            # dcB/dcB
            if j == Nz:
                if i == Nz:
                    Jac[j,i] = (-1/dz)*(v + D/dz) - 2*k0*exp(-E/T[i-Nz])*cA[i-Nz]
                elif i == (Nz+1):
                    Jac[j,i] = D/dz**2
            elif Nz < j < (2*Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) - 2*k0*exp(-E/T[i-Nz])*cA[i-Nz]
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
            elif j == (2*Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) - 2*k0*exp(-E/T[Nz-1])*cA[Nz-1]
            
            # dcB/dT
            if Nz<=j<(2*Nz) and i == (j+Nz):
                Jac[j,i] = - 2*k0*(-E/T[i-2*Nz])*exp(-E/T[i-2*Nz])*cA[i-2*Nz]*cB[i-2*Nz]
            
                
            # dT/dcA
            if (2*Nz)<=j and i==(j-2*Nz):
                Jac[j,i] = B*k0*exp(-E/T[i])*cB[i]
                
            # dT/dcB 
            if (2*Nz)<=j and i==(j-Nz):
                Jac[j,i] = B*k0*exp(-E/T[i-Nz])*cA[i-Nz]

            # dT/dT
            if j == (2*Nz):
                if i == (2*Nz):
                    Jac[j,i] = (-1/dz)*(v + D/dz) + B*k0*(-E/T[i-2*Nz])*exp(-E/T[i-2*Nz])*cA[i-2*Nz]*cB[i-2*Nz]
                elif i == (2*Nz + 1):
                    Jac[j,i] = D/dz**2
            elif (2*Nz) < j < (3*Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) + B*k0*(-E/T[i-2*Nz])*exp(-E/T[i-2*Nz])*cA[i-2*Nz]*cB[i-2*Nz]
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
            elif j == (3*Nz-1):
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
    cA = np.array(x[:Nz])
    cB = np.array(x[Nz:2*Nz])
    T  = np.array(x[2*Nz:])
    

    '''
    ### Implementation of compound A
    '''
    # Advection of compound A
    NadvA     = np.zeros([Nz+1])
    NadvA[0]  = v*cAin
    NadvA[1:] = v*cA[:]
    
    # Diffusion of compound A
    JA       = np.zeros([Nz+1])
    JA[1:Nz] = (-D/dz) *  (cA[1:] - cA[:Nz-1])
    
    # Flux = advection + diffusion for compound A
    NA = NadvA + JA

    # Production rate of A
    r  = k0*exp(-E/T)*cA*cB
    RA = -r
    
    # Advection-Diffusion-Reaction Equation for Compound A
    cAdot = (NA[1:] - NA[:Nz])/(-dz) + RA
    
    '''
    ### Implementation of compound B
    '''
    # Advection of compound B
    NadvB     = np.zeros([Nz+1])
    NadvB[0]  = v*cBin
    NadvB[1:] = v*cB[:]
    
    # Diffusion of compound B
    JB       = np.zeros([Nz+1])
    JB[1:Nz] = (-D/dz) *  (cB[1:] - cB[:Nz-1])
    
    # Flux = advection + diffusion for compound B
    NB = NadvB + JB

    # Production rate of B
    r  = 2*k0*exp(-E/T)*cA*cB
    RB = -r
    
    # Advection-Diffusion-Reaction Equation for Compound B
    cBdot = (NB[1:] - NB[:Nz])/(-dz) + RB
    
    '''
    ### Implementation of heat T
    '''
    # Advection of heat T
    NadvT     = np.zeros([Nz+1])
    NadvT[0]  = v*Tin
    NadvT[1:] = v*T[:]
    
    # Diffusion of heat T
    JT       = np.zeros([Nz+1])
    JT[1:Nz] = (-D/dz) *  (T[1:] - T[:Nz-1])
    
    # Flux = advection + diffusion for heat T
    NT = NadvT + JT

    # Production rate of T
    r  = B*k0 * exp(-E/T) * cA * cB
    RT = r
    
    # Advection-Diffusion-Reaction Equation for heat T
    Tdot = (NT[1:] - NT[:Nz])/(-dz) + RT
    
    '''
    ### Formatting output
    '''
    cAdot = np.array([cAdot]).T
    cBdot = np.array([cBdot]).T
    Tdot  = np.array([ Tdot]).T
    
    xdot = np.row_stack([cAdot, cBdot, Tdot])
    xdot = xdot[:,0]
    
    # Initialize Jacobian
    N = 3*Nz
    Jac = np.zeros([N,N])
    
    for j in range(N): # Row
        for i in range(N): # Column
        
            # dcA/dcA
            if j == 0:
                if i == 0:
                    Jac[j,i] = (-1/dz)*(v + D/dz) - k0*exp(-E/T[i])*cB[i]
                elif i == 1:
                    Jac[j,i] = D/dz**2
                    
            elif 0 < j <(Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) - k0*exp(-E/T[j])*cB[j]
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
                    
            elif j == (Nz-1):
                if i == (Nz-2):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == (Nz-1):
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) - k0*exp(-E/T[j])*cB[j]

            # dcA/dcB
            if 0<=j<Nz and i==(Nz+j):
                Jac[j,i] = - k0*exp(-E/T[j])*cA[j]

            # dcA/dT
            if 0<=j<Nz and i==(2*Nz+j):
                Jac[j,i] = - k0*(-E/T[j])*exp(-E/T[j])*cA[j]*cB[j]
                
            # dcB/dcA
            if Nz<=j<(2*Nz) and i==(j-Nz):
                Jac[j,i] = -2*k0*exp(-E/T[i])*cB[i]
                
            # dcB/dcB
            if j == Nz:
                if i == Nz:
                    Jac[j,i] = (-1/dz)*(v + D/dz) - 2*k0*exp(-E/T[i-Nz])*cA[i-Nz]
                elif i == (Nz+1):
                    Jac[j,i] = D/dz**2
            elif Nz < j < (2*Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) - 2*k0*exp(-E/T[i-Nz])*cA[i-Nz]
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
            elif j == (2*Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) - 2*k0*exp(-E/T[Nz-1])*cA[Nz-1]
            
            # dcB/dT
            if Nz<=j<(2*Nz) and i == (j+Nz):
                Jac[j,i] = - 2*k0*(-E/T[i-2*Nz])*exp(-E/T[i-2*Nz])*cA[i-2*Nz]*cB[i-2*Nz]
            
                
            # dT/dcA
            if (2*Nz)<=j and i==(j-2*Nz):
                Jac[j,i] = B*k0*exp(-E/T[i])*cB[i]
                
            # dT/dcB 
            if (2*Nz)<=j and i==(j-Nz):
                Jac[j,i] = B*k0*exp(-E/T[i-Nz])*cA[i-Nz]

            # dT/dT
            if j == (2*Nz):
                if i == (2*Nz):
                    Jac[j,i] = (-1/dz)*(v + D/dz) + B*k0*(-E/T[i-2*Nz])*exp(-E/T[i-2*Nz])*cA[i-2*Nz]*cB[i-2*Nz]
                elif i == (2*Nz + 1):
                    Jac[j,i] = D/dz**2
            elif (2*Nz) < j < (3*Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) + B*k0*(-E/T[i-2*Nz])*exp(-E/T[i-2*Nz])*cA[i-2*Nz]*cB[i-2*Nz]
                elif i == (j+1):
                    Jac[j,i] = D/dz**2
            elif j == (3*Nz-1):
                if i == (j-1):
                    Jac[j,i] = (1/dz)*(v + D/dz)
                elif i == j:
                    Jac[j,i] = (-1/dz)*(v + 2*D/dz) + B*k0*(-E/T[Nz-1])*exp(-E/T[Nz-1])*cA[Nz-1]*cB[Nz-1]
                    
    return xdot, Jac