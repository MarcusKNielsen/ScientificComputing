from numpy import zeros, size, array, row_stack
import numpy as np



def Solve(Model, tspan, x0, h0, abstol, reltol, args):
    t = tspan[0]            # Initial time
    tf = tspan[1]           # Final time
    dt = h0                 # Initial time step
    x = x0                  # initial state
    
    # Error Controller Parameters
    epstol = 0.8            # Target
    facmin = 0              # Minimum allowed step factor
    facmax = np.inf            # Maximum allowed step factor
    
    
    # Allocation of space for solution
    X = array([x])          # States as a function of time
    T = array([t])          # Time
    R = array([0])
    H = array([dt])
    
    # Function evaluation counter
    Nfev = 0
    Naccept = 0
    Nreject = 0
    
    while t < tf:
        
        if t+dt > tf:
            dt = tf - t
        
        Nfev += 1
        f = Model(t, x, *args)
        
        AcceptStep = False
        while not AcceptStep:
            
            # Calculate the next x using a full dt step.
            xnxt = x + dt*f
        
            # Calculate the next x using two half steps.
            dth = 0.5*dt                    # dt half
            x1nxt = x + dth*f
            th = t + dth                    # t plus a half step
            Nfev += 1
            f = Model(th, x1nxt, *args)     # Evaluate the model in the calculated half step.
            x2nxt = x1nxt + dth*f 
        
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
                
                #E = row_stack((E,abs(e)))
                R = np.append(R,r)
                H = np.append(H,dt)
                
            elif AcceptStep == False:
                Nreject += 1
                
                
            # Step size controller
            dt = max(facmin,min(np.sqrt(epstol/r),facmax))*dt
    
    return T,X,Nfev,R,H,Naccept,Nreject
