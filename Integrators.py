import numpy as np

#Algoritmo de Rubens
def Rubens(coord, vel, dt, tiempo, n, mass = None):
    E = 0.5
    a = E*dt
    c0 = np.exp(-a)
    c1 = (1 - c0)/a
    c2 = (1 - c1)/a
    m = 1
    fj = np.random.rand(len(tiempo), n)
    xj = np.random.rand(len(tiempo), n)
    vxj = np.random.rand(len(tiempo), n)
    for i in range(len(tiempo)-1):
        coord[i+1] = coord[i] + c1*vel[i]*dt + (c2*fj[i]*dt**2)/m + xj[i]
        vel[i+1] = c0*vel[i] + ((c1-c2)*fj[i]*dt)/m + (c2*fj[i+1]*dt)/m + vxj[i]
    return coord, vel

#Algoritmo de Beeman
def Beeman(coord, vel, dt, tiempo, n, mass):
    
     # Aceleraciones aleatorias
    ac = np.random.uniform(-1e2,1e2,(len(tiempo), n))
    # Agregar efecto de la masa sobre la aceleración de cada partícula
    for i in range(n):
        ac[:,i] = ac[:,i]/mass[i]
        
    for i in range (len(tiempo)-1):
        coord[i+1]=coord[i]+vel[i]*(dt)+(2/3)*ac[i]*(dt**2)-(1/6)*ac[i-1]*(dt**2)
        vel[i+1]=vel[i]+(1/3)*ac[i+1]*dt+(5/6)*ac[i]*dt-(1/6)*ac[i-1]*dt
    return coord, vel

# Integrador Velocidades de Verlet
def Velocity_Verlet(coord, vel, dt, tiempo, n, mass):
    
     # Aceleraciones aleatorias
    ac = np.random.uniform(-1e2,1e2,(len(tiempo), n))
    # Agregar efecto de la masa sobre la aceleración de cada partícula
    for i in range(n):
        ac[:,i] = ac[:,i]/mass[i]
        
    for i in range(len(tiempo) - 1):
        coord[i+1] = coord[i] + vel[i]*dt+1/2*ac[i]*dt**2
        vel[i+1] = vel[i]  + (1/2)*(ac[i]+ac[i+1])*dt
    return coord, vel

def Verlet(coord, vel, dt, tiempo, n, mass = None):
     # Aceleraciones aleatorias
    fj = np.random.uniform(-1e2,1e2,(len(tiempo), n))
    # Agregar efecto de la masa sobre la aceleración de cada partícula
    for i in range(n):
        fj[:,i] = fj[:,i]/mass[i]
    coord[1] = coord[0] + (fj[0]*dt**2)/2
    for i in range(1, len(tiempo)-1):
        coord[i+1] = 2*coord[i] - coord[i-1] + fj[i]*dt**2
    vel = np.diff(coord)/dt
    return coord, vel

def LeapFrog(coord, vel, dt, tiempo, n, mass):
     # Aceleraciones aleatorias
    fj = np.random.uniform(-1e2,1e2,(len(tiempo), n))
    # Agregar efecto de la masa sobre la aceleración de cada partícula
    for i in range(n):
        fj[:,i] = fj[:,i]/mass[i]
    # Agregar efecto de la masa sobre la aceleración de cada partícula
    for i in range(n):
        fj[:,i] = fj[:,i]/mass[i]
    for i in range(len(tiempo)-1):
        vel[i + 1] = vel[i] - coord[i]/dt + fj[i]*dt
        coord[i + 1] = coord[i] + vel[i + 1]*dt
    return coord, vel

def Langevin(coord , vel, dt, tiempo, n, mass = None):
    chi = np.random.uniform(-1, 1, len(tiempo))
    theta = np.random.uniform(-1, 1, len(tiempo))
    gamma = sigma = 1
    for i in range(len(tiempo) - 1):
        fj = np.random.uniform(-1e2,1e2, n)
        Cj = (0.5*dt**2)*(fj - gamma*vel[i]) + sigma *(dt **(3/2)) * (0.5*chi[i]+(1/(2*np.sqrt(3)))* theta[i])
        coord[i+1] = coord[i] + dt*vel[i] + Cj
        fj1 = np.random.uniform(-1e2,1e2,n)
        vel[i+1] = vel[i] + (0.5*dt)*(fj1 + fj) - dt*gamma*vel[i] + sigma*np.sqrt(dt)*chi[i] - gamma*Cj
    return coord, vel

def Brownian(coord, vel, dt, tiempo, n, mass = None):
    delta_t = np.sqrt(dt)
    coord[1:] = delta_t * np.random.normal(0, 1, size=(len(tiempo)-1, n))
    coord = np.cumsum(coord, axis=0)
    vel = np.diff(coord)/dt
    return coord, vel