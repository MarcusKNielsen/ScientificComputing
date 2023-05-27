import numpy as np


def Model(t,x,p):
    
    # Unpack parameter vector
    u = p
    
    # Unpack state vector
    x1,x2 = x
    
    # We define the RHS f(x,t) = (f1, f2)
    f1 = x2
    f2 = u*(1-x1**2)*x2 - x1
    
    # The ODE system
    xdot = np.array([f1,f2])
    
    return xdot
    

def Jacobian(t,x,p):
    
    # Unpack parameter vector
    u = p
    
    # Unpack state vector
    x1,x2 = x
    
    # Define the Jacobian of f
    J11 = 0                  # df1/dx1
    J12 = 1                  # df1/dx2
    J21 = -2*u*x1*x2 - 1     # df2/dx1
    J22 = u*(1-x1**2)         # df2/dx2
    Jac = np.array([[J11, J12],[J21, J22]]) # The Jacobian
    
    return Jac

def ModelJacobian(t,x,p):
    
    # Unpack parameter vector
    u = p
    
    # Unpack state vector
    x1,x2 = x
    
    # We define the RHS f(x,t) = (f1, f2)
    f1 = x2
    f2 = u*(1-x1**2)*x2 - x1
    
    # The ODE system
    xdot = np.array([f1,f2])
    
    # Define the Jacobian of f
    J11 = 0                  # df1/dx1
    J12 = 1                  # df1/dx2
    J21 = -2*u*x1*x2 - 1     # df2/dx1
    J22 = u*(1-x1**2)         # df2/dx2
    Jac = np.array([[J11, J12],[J21, J22]]) # The Jacobian
    
    return np.array([xdot,Jac])


## Stochastic Version of the Van der Pol Problem:

# Here it is assumed that the p vector has two entries
# the mu or u paramter which is the first entry in p that is: u=p[0]
# the second entry is the scale of the diffusion part sigma: sigma=p[1]  

def drift(t,x,p):
    # Unpack parameter vector
    u = p[0]
    
    # Unpack state vector
    x1,x2 = x
    
    # We define the RHS f(x,t) = (f1, f2)
    f1 = x2
    f2 = u*(1-x1**2)*x2 - x1
    
    return np.array([f1,f2])
    
def diffusion(t,x,p):
    sigma = p[1]
    g = np.array([0.0,sigma])
    return g

def JacobianSDE(t,x,p):
    p = p[0]
    return Jacobian(t,x,p)



    

