from numpy import exp
import numpy as np


def Model(t,x,p):
    
    # Unpack parameter vector
    F,V,k0,E,B,cAin,cBin,Tin= p
    
    # Unpack state vector
    T = x
    
    # We define the RHS f(x,t) = f1
    
    cA = cAin +   (Tin - T)/B
    cB = cBin + 2*(Tin - T)/B
    
    f1 = (F/V) * (Tin - T) + B * k0 * exp(-E/T) * cA * cB

    # The ODE system
    xdot = f1
    
    return xdot



def Jacobian(t,x,p):
    
    # Unpack parameter vector
    F,V,k0,E,B,cAin,cBin,Tin= p
    
    # Unpack state vector
    T = x

    # The Jacobian
    
    cA = cAin +   (Tin - T)/B
    cB = cBin + 2*(Tin - T)/B
    
    Jac = np.array([(-F/V) + B*k0*exp(-E/T)*( (E*cA*cB/T**2) - (cB + 2*cA)/B )])
    
    return Jac


def ModelJacobian(t,x,p):
    
    # Unpack parameter vector
    F,V,k0,E,B,cAin,cBin,Tin= p
    
    # Unpack state vector
    T = x
    
    # We define the RHS f(x,t) = f1
    
    cA = cAin + (Tin - T)/B
    cB = cBin + (Tin - T)/B
    
    f1 = (F/V) * (Tin - T) + B * k0 * exp(-E/T) * cA * cB

    # The ODE system
    xdot = f1
    
    # The Jacobian
    
    cA = cAin +   (Tin - T)/B
    cB = cBin + 2*(Tin - T)/B
    
    Jac = np.array([(-F/V) + B*k0*exp(-E/T)*( (E*cA*cB/T**2) - (cB + 2*cA)/B )])
    
    return np.array([xdot,Jac])