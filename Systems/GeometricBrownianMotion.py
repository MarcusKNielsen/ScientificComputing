

def drift(t,x,p):
    u = p[0]
    return u*x

def diffusion(t,x,p):
    sigma = p[1]
    return sigma*x

def Jacobian(t,x,p):
    u = p[0]
    return u