import numpy as np
#from NewtonsMethodODE_Modified2 import NewtonsMethodODE_Modified as NewtonsMethod
from InexactNewtonsMethod import NewtonsMethodODE_Modified as NewtonsMethod


def Solve( Drift, Diffusion, JacobianDrift, tspan, x0, W, args):
    
    Ns    = len(W[0,0,:])       # Number of Simulations
    N     = len(W[:,0,0])       # Number of time points needed
    nx    = len(x0)             # Dimension of State vector
    X     = np.zeros([N,nx,Ns]) # State array
    T     = np.zeros(N)         # Time array
    t0    = tspan[0]            # Initial Time
    tN    = tspan[1]            # Final time
    dt    = (tN - t0)/N         # Time step size
    T[0]  = t0
    
    # Parameter for Newtons Method
    TOL = 10**(-8)          # Tolerance
    MaxIter = 100           # Maximum number of iterations
    
    
    for i in range(Ns):
        X[0,:,i] = x0
        
        Nfev = 0
        Njac = 0
        Nlu  = 0
        
        for k in range(N-1):
            
            # Current time step and state
            tnxt  = T[k] + dt
            xk = X[k,:,i]
            
            # Diffusion element 
            g = Diffusion(tnxt, xk, *args)
            dW = W[k+1,:,i] - W[k,:,i]
            c = xk + g*dW

            # Append solution
            T[k+1]     = tnxt
            xinit = xk + dt*Drift(tnxt,xk,*args)
            X[k+1,:,i], nfev, njac, nlu = NewtonsMethod(Drift, JacobianDrift,tnxt,dt,xinit,c, TOL, MaxIter, *args)

            Nfev += nfev + 2
            Njac += njac
            Nlu  += nlu

        print(f"Nfev = {Nfev}")
        print(f"Njac = {Njac}")
        print(f"Nlu = {Nlu}")

    return T, X, Nfev, Njac, Nlu



