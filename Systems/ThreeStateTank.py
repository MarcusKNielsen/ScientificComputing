from numpy import exp
import numpy as np

def Model(t,x,p):
    
    # Unpack parameter vector
    F,V,k0,E,B,cAin,cBin,Tin= p
    
    # Unpack state vector
    cA, cB, T = x
    
    # We define the RHS f(x,t) = (f1, f2, f3)
    f1 = (F/V) * (cAin - cA) -   k0 * exp(-E/T) * cA * cB
    f2 = (F/V) * (cBin - cB) - 2*k0 * exp(-E/T) * cA * cB
    f3 = (F/V) * (Tin  -  T) + B*k0 * exp(-E/T) * cA * cB
    
    # The ODE system
    xdot = np.array([f1,f2,f3])
    
    return xdot


def Jacobian(t,x,p):
    
    # Unpack parameter vector
    F,V,k0,E,B,cAin,cBin,Tin= p
    
    # Unpack state vector
    cA, cB, T = x
    
    # Define the Jacobian of f
    J11 = (-F/V) -              k0 * exp(-E/T)      * cB   # df1/dx1
    J12 =        -              k0 * exp(-E/T) * cA        # df1/dx2
    J13 =        -   (E/T**2) * k0 * exp(-E/T) * cA * cB   # df1/dx3
    
    J21 =        - 2          * k0 * exp(-E/T)      * cB   # df2/dx1
    J22 = (-F/V) - 2          * k0 * exp(-E/T) * cA        # df2/dx2
    J23 =        - 2*(E/T**2) * k0 * exp(-E/T) * cA * cB   # df2/dx3
    
    J31 =        + B          * k0 * exp(-E/T)      * cB   # df3/dx1
    J32 =        + B          * k0 * exp(-E/T) * cA        # df3/dx2
    J33 = (-F/V) + B*(E/T**2) * k0 * exp(-E/T) * cA * cB   # df3/dx3

    # The Jacobian
    Jac = np.array([[J11, J12, J13],
                    [J21, J22, J23],
                    [J31, J32, J33]]) 
    
    return Jac


def ModelJacobian(t,x,p):
    
    # Unpack parameter vector
    F,V,k0,E,B,cAin,cBin,Tin= p
    
    # Unpack state vector
    cA, cB, T = x
    
    # We define the RHS f(x,t) = (f1, f2, f3)
    f1 = (F/V) * (cAin - cA) -   k0 * exp(-E/T) * cA * cB
    f2 = (F/V) * (cBin - cB) - 2*k0 * exp(-E/T) * cA * cB
    f3 = (F/V) * (Tin  -  T) + B*k0 * exp(-E/T) * cA * cB
    
    # The ODE system
    xdot = np.array([f1,f2,f3])
    
    # Define the Jacobian of f
    J11 = (-F/V) -              k0 * exp(-E/T)      * cB   # df1/dx1
    J12 =        -              k0 * exp(-E/T) * cA        # df1/dx2
    J13 =        -   (E/T**2) * k0 * exp(-E/T) * cA * cB   # df1/dx3
    
    J21 =        - 2          * k0 * exp(-E/T)      * cB   # df2/dx1
    J22 = (-F/V) - 2          * k0 * exp(-E/T) * cA        # df2/dx2
    J23 =        - 2*(E/T**2) * k0 * exp(-E/T) * cA * cB   # df2/dx3
    
    J31 =        + B          * k0 * exp(-E/T)      * cB   # df3/dx1
    J32 =        + B          * k0 * exp(-E/T) * cA        # df3/dx2
    J33 = (-F/V) + B*(E/T**2) * k0 * exp(-E/T) * cA * cB   # df3/dx3

    # The Jacobian
    Jac = np.array([[J11, J12, J13],
                    [J21, J22, J23],
                    [J31, J32, J33]])
    
    return np.array([xdot,Jac])