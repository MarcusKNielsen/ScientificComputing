from numpy import exp, pi

def Model(t,x,p):
    a,b,k,u = p
    return 4*a*b*(0.1/x)*(k-x)/k**2 + a*(u-b)

def drift(t,x,p):
#    k,sigma = p
#    return exp( -(1/x)-(1/(k-x)) )
    A,B,k,a,u,sigma = p
    return ((A/x) - (B/(k-x))) - a*u

def diffusion(t,x,p):
    A,B,k,a,u,sigma = p
#    return sigma*exp( -(1/x)-(1/(k-x)) )
    return sigma*x**0.1*exp(-1.5*x)
