import numpy as np # linear algebra
import math
import matplotlib.pyplot as plt

# 1D FDTD, 30.48 cm thick slab, mu = 2.0, epsilon = 6.0, calculate reflectance and transmittance over 0 to 1 GHz

# Physical constants (units in meters, seconds)
c0 = 299792458
gigahertz = 1e9
epsilon = 6.0
mu = 2.0
nmax = np.sqrt(mu*epsilon)
fmax = 1*gigahertz
# nbc REFRACTIVE INDEX ABOUT THE TWO BOUNDARIES
nbc = 1.0
Nz = 94
pi = math.pi

# grid resolution
dz = 0.4293e-2
# device on grid
UR = np.ones(Nz)
ER = np.ones(Nz)
UR[13:84] = mu
ER[13:84] = epsilon
nsrc = np.sqrt(UR*ER)

# INITIALISE FDTD
# time step
dt = nbc*dz/(2*c0)
# source parameters
n_source = 2
tau = 0.5/fmax
t0 = 6*tau
tprop = nmax*Nz*dz/(c0)
T = 12*tau + 5*tprop
steps = int(np.ceil(T/dt))

# Compute Gaussian source parameters
time = np.array(range(steps))*dt
# total delay between E and H
delt = 0.5*nsrc[2]*dz/c0 + 0.5*dt
# amplitude of H field
A = -np.sqrt(ER/UR)

# E and H field sources
Esrc = np.exp(-((time-t0)/tau)**2)
Hsrc = A[1]*np.exp(-((time-t0+delt)/tau)**2)

# INITIALISE FOURIER TRANSFORMS
Nfreq = 100
freq = np.array(np.linspace(0, int(fmax), Nfreq))
K = np.exp(-1j*2*pi*dt*freq)
EyR = np.zeros(Nfreq, dtype=complex)
EyT = np.zeros(Nfreq, dtype=complex)
SRC = np.zeros(Nfreq, dtype=complex)

# COMPUTE UPDATE COEFICIENTS
mHx = c0 * dt / UR
mEy = c0 * dt / ER

# INITIALISE FILEDS TO ZERO
Ey = np.zeros(Nz)
Hx = np.zeros(Nz)

# INITIALISE BOUNDARY TERMS TO ZERO
h1, h2, e1, e2 = 0, 0, 0, 0

# Main FDTD Loop
for t in range(steps):

    h2 = h1
    h1 = Hx[0]

    # UPDATE H FROM E
    Hx[0:Nz-1] += mHx[0:Nz-1] * (Ey[1:Nz] - Ey[0:Nz-1])/dz
    Hx[Nz-1] += mHx[Nz-1] * (e2 - Ey[Nz-1])/dz

    # HANDLE H AT SOURCE POINT
    Hx[n_source-1] -= (mHx[n_source-1]/dz) * Esrc[t]

    e2 = e1
    e1 = Ey[Nz-1]

    # UPDATE E FROM H
    Ey[1:Nz] += mEy[1:Nz] * (Hx[1:Nz] - Hx[0:Nz-1])/dz
    Ey[0] += mEy[0] * (Hx[0] - h2)/dz

    # HANDLE E AT SOURCE POINT
    Ey[n_source] -= (mEy[n_source]/dz) * Hsrc[t]

    # UPDATE FOURIER TRANSFORMS
    EyR[0:Nfreq] += (K[0:Nfreq]**(t+1))*Ey[0]
    EyT[0:Nfreq] += (K[0:Nfreq]**(t+1))*Ey[Nz-1]
    SRC[0:Nfreq] += (K[0:Nfreq]**(t+1))*Esrc[t]

# NORMALISE REFLECTANCE AND TRANSMITTANCE
REF = abs(EyR/SRC)**2
TRA = abs(EyT/SRC)**2
CON = REF + TRA

plt.plot(freq, CON)
plt.plot(freq, REF)
plt.plot(freq, TRA)
plt.xlabel('Frequency (Hz)')
plt.grid()
plt.show()
