from numpy import zeros,size, array, row_stack
import numpy as np


def DP54(Model, xn, tn, h, args):
    
    # First we define all coefficients in the Butcher Tableau.
    # Coefficients which are zero are skipped.
    
    # We define the time coefficients
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    c7 = 1.0
    
    # The A matrix coefficients:
    a21 =  1/5
    a31 =  3/40
    a32 =  9/40
    
    a41 =  44/45
    a42 = -56/15
    a43 =  32/9
    
    a51 =  19372/6561
    a52 = -25360/2187
    a53 =  64448/6561
    a54 = -212/729
    
    a61 =  9017/3168 
    a62 = -355/33
    a63 =  46732/5247
    a64 =  49/176
    a65 = -5103/18656
    
    a71 =  35/384
    a73 =  500/1113
    a74 =  125/192
    a75 = -2187/6784
    a76 =  11/84
    
    # The b vector
    b1 =  5179/57600
    b3 =  7571/16695
    b4 =  393/640
    b5 = -92097/339200
    b6 =  187/2100
    b7 =  1/40
    
    # The d vector
    d1 =  71/57600
    d3 = -71/16695
    d4 =  71/1920
    d5 = -17253/339200
    d6 =  22/525
    d7 = -1/40
    
    # We define the 7 time stages:
    T1 = tn
    T2 = tn + c2 * h
    T3 = tn + c3 * h
    T4 = tn + c4 * h
    T5 = tn + c5 * h
    T6 = tn + c6 * h
    T7 = tn + c7 * h
    
    # We calculate the 7 intermediate stages:
    X1 = xn
    k1 = Model(T1,X1,args)
    
    X2 = xn + h * ( a21 * k1 )
    k2 = Model(T2,X2,args)
    
    X3 = xn + h * ( a31 * k1 + a32 * k2 )
    k3 = Model(T3,X3,args)
    
    X4 = xn + h * ( a41 * k1 + a42 * k2 + a43 * k3 )
    k4 = Model(T4,X4,args)
    
    X5 = xn + h * ( a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4 )
    k5 = Model(T5,X5,args)
    
    X6 = xn + h * ( a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5 )
    k6 = Model(T6,X6,args)
    
    X7 = xn + h * ( a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6 )
    k7 = Model(T7,X7,args)
    
    # Calculate next step (xnxt) and next error (enxt)
    
    xnxt = xn + h * ( b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 )
    
    enxt = h * ( d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7 )
    
    return xnxt, enxt



def Solve(Model, tspan, x0, h0, abstol, reltol, args):

    t = tspan[0]            # Initial time
    tf = tspan[1]           # Final time
    dt = h0                 # Initial time step
    x = x0                  # initial state

    # Error Controller Parameters
    epstol = 0.8            # Target
    facmin = 0.01          # Minimum allowed step factor
    facmax = 1.0            # Maximum allowed step factor

    # Allocation of space for solution
    X = array([x])          # States as a function of time
    T = array([t])          # Time
    
    while t < tf:
        
        if t+dt > tf:
            dt = tf - t

        AcceptStep = False
        while not AcceptStep:
            
            xnxt, enxt = DP54(Model, x, t, dt, *args) 

            r = np.max(np.abs(enxt) / np.maximum(abstol, np.abs(xnxt) * reltol))
        
            AcceptStep = (r <= 1)
            if AcceptStep:
                
                t = t + dt
                x = xnxt
                
                X = row_stack((X, x))
                T = np.append(T,t)
                
            # Step size controller
            dt = max(facmin,min((epstol/r)**(0.2),facmax))*dt
    
    return T,X



