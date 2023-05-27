import numpy as np


# The system of differential equations.
# Here X1 is the number of prey
# and X2 is the number of predators.

def Model(t,x,p): 
    
    # Unpack parameter vector
    a,b,u,v = p
    
    # Unpack state vector
    x1,x2 = x
    
    # We define the RHS f(x,t) = (f1, f2)
    f1 = a*x1 - b*x1*x2
    f2 = u*x1*x2 - v*x2
    
    # The ODE system
    xdot = np.array([f1,f2])
    
    return xdot
    

# The Jacobian of the differential equation model above.

def Jacobian(t,x,p):
    
    # Unpack parameter vector
    a,b,u,v = p
    
    # Unpack state vector
    x1,x2 = x
    
    # Define the Jacobian of f
    J11 = a - b*x2      # df1/dx1
    J12 = - b*x1        # df1/dx2
    J21 = u*x2          # df2/dx1
    J22 = u*x1 - v      # df2/dx2
    Jac = np.array([[J11, J12],[J21, J22]]) # The Jacobian
    
    return Jac

def ModelJacobian(t,x,p):
    
    # Unpack parameter vector
    a,b,u,v = p
    
    # Unpack state vector
    x1,x2 = x
    
    # We define the RHS f(x,t) = (f1, f2)
    f1 = a*x1 - b*x1*x2
    f2 = u*x1*x2 - v*x2
    
    # The ODE system
    xdot = np.array([f1,f2])

    # Define the Jacobian of f
    J11 = a - b*x2      # df1/dx1
    J12 = - b*x1        # df1/dx2
    J21 = u*x2          # df2/dx1
    J22 = u*x1 - v      # df2/dx2
    Jac = np.array([[J11, J12],[J21, J22]]) # The Jacobian
    
    return np.array([xdot,Jac])


## Stochastic Version of the PreyPredator Model:

    
def Drift(t,x,p):

    
    # Unpack parameter vector
    a,b,u,v = p[:4]
    
    # Unpack state vector
    x1,x2 = x
    
    # We define the drift of the system
    f1 = a*x1 - b*x1*x2
    f2 = u*x1*x2 - v*x2
    
    # The ODE system
    f = np.array([f1,f2])
    
    return f
    
    
def Diffusion(t,x,p):
    
    # Unpack parameter vector
    sigma1, sigma2 = p[-2:]
    
    # Unpack state vector
    x1,x2 = x
    
    # We define the diffusion of the system
    g = np.array([sigma1*x1,sigma2*x2])
    
    return g


def JacobianDrift(t,x,p):
    
    # Unpack parameter vector
    a,b,u,v = p[:4]
    
    # Unpack state vector
    x1,x2 = x
    
    # Define the Jacobian of f
    J11 = a - b*x2      # df1/dx1
    J12 = - b*x1        # df1/dx2
    J21 = u*x2          # df2/dx1
    J22 = u*x1 - v      # df2/dx2
    
    # The Jacobian
    Jac = np.array([[J11, J12],[J21, J22]]) 
    
    return Jac
    



