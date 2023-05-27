import numpy as np


def Solve( Drift, Diffusion, tspan, x0, W, args):
    
    Ns    = len(W[0,0,:])       # Number of Simulations
    N     = len(W[:,0,0])       # Number of time points needed
    nx    = len(x0)             # Dimension of State vector
    X     = np.zeros([N,nx,Ns]) # State array
    T     = np.zeros(N)         # Time array
    t0    = tspan[0]            # Initial Time
    tN    = tspan[1]            # Final time
    dt    = (tN - t0)/N         # Time step size
    T[0]  = t0
    Nfev  = 0
    
    for i in range(Ns):
        X[0,:,i] = x0
        Nfev  = 0
        for k in range(N-1):
            
            # Current time step and state
            t = T[k] + dt
            x = X[k,:,i]
            
            
            f  = Drift(t, x, *args)
            g  = Diffusion(t, x, *args)
            dW = W[k+1,:,i] - W[k,:,i]
            

            # Append solution
            T[k+1]     = t
            X[k+1,:,i] = x + f*dt + g*dW
            
            Nfev += 2
        print(Nfev)

    return T, X, Nfev