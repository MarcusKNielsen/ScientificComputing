from numpy import array, row_stack
#from NewtonsMethodODE_Modified2 import NewtonsMethodODE_Modified as NewtonsMethod
from InexactNewtonsMethod import NewtonsMethodODE_Modified as NewtonsMethod
import numpy as np


def Solve(Model, Jacobian, tspan, x0, h0, abstol, reltol, args):
    t = tspan[0]            # Initial time
    tf = tspan[1]           # Final time
    dt = h0                 # Initial time step
    x = x0                  # initial state
    
    # Error Controller Parameters
    epstol = 0.8            # Target
    facmin = 0              # Minimum allowed step factor
    facmax = np.inf         # Maximum allowed step factor
    
    
    # Allocation of space for solution
    X = array([x])          # States as a function of time
    T = array([t])          # Time
    R = array([0])
    H = array([dt])
    
    # Parameter for Newtons Method
    TOL = 10**(-6)          # Tolerance
    MaxIter = 100           # Maximum number of iterations
 
    # Function evaluation counter
    Nfev    = 0
    Njac    = 0
    Nlu     = 0
    Naccept = 0
    Nreject = 0
    
    while t < tf:
        
        if t+dt > tf:
            dt = tf - t
        
        AcceptStep = False
        while not AcceptStep:
            
            # Calculate time steps
            dth  = 0.5*dt           # half dt
            th   = t +  dth         # half evaluation point
            tnxt = t + dt           # full evaluation point
            
            # Calculate the next x using a full dt step.
            xinit = x + dt*Model(t, x, *args)
            xnxt, nfev, njac, nlu = NewtonsMethod(Model, Jacobian, tnxt, dt, xinit, x, TOL, MaxIter, *args)
            Nfev += nfev + 1
            Njac += njac
            Nlu += nlu
            
            # Calculate the next x using two half steps.
            xinit = x + dth*Model(t, x, *args)
            x1nxt, nfev, njac, nlu = NewtonsMethod(Model, Jacobian, th, dth, xinit, x, TOL, MaxIter, *args)
            Nfev += nfev + 1
            Njac += njac
            Nlu += nlu
            
            xinit = x1nxt + dth*Model(t, x1nxt, *args)
            x2nxt, nfev, njac, nlu = NewtonsMethod(Model, Jacobian, tnxt, dth, xinit, x1nxt, TOL, MaxIter, *args)
            Nfev += nfev + 1
            Njac += njac
            Nlu += nlu
        
            # Error calculate
            e = x2nxt - xnxt
            
            r = np.max(np.abs(e) / np.maximum(abstol, np.abs(x2nxt) * reltol))
        
            AcceptStep = (r <= 1)
            if AcceptStep:
                Naccept += 1
                t = t + dt
                x = x2nxt
                
                X = row_stack((X, x))
                T = np.append(T,t)
                
                R = np.append(R,r)
                H = np.append(H,dt)
                
            elif AcceptStep == False:
                Nreject += 1
                
            # Step size controller
            dt = max(facmin,min(np.sqrt(epstol/r),facmax))*dt
    
    return T, X, Nfev, Njac, Nlu, R, H, Naccept, Nreject


