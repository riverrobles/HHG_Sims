# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:22:59 2017

@author: jmann
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import scipy.integrate as spin
import scipy as sp
from scipy import signal
import scipy.fftpack
import os
import time
from sys import platform
if platform == 'win32' or platform == 'cygwin':
    import tkinter as tk
    from tkinter import filedialog
    from matplotlib.widgets import SpanSelector
from multiprocessing import Pool, TimeoutError
import matplotlib.gridspec as gridspec
import glob
from matplotlib.widgets import Slider


hbar = 1.054571800e-34 #J s
me = 9.10938356e-31 #kg
eV_J = 1.602e-19 
c = 3e8
qe = 1.602e-19
e0 = 8.854e-12
a0 = 5.2917721092e-11
auE = 4.35974417e-18

PEcmap = 'YlGnBu' #Colormap for electron and photon time dependent spectra

def waveletTransform(sig, wav, wid, ts, newLen):
    N = len(sig)
    newTs = np.linspace(ts[0], ts[-1], newLen)
#    res = []
#    for i in wid:
#        f = wav(N, i)
#        res.append(np.interp(newTs, ts, signal.convolve(sig, f, mode="same")))
#    return np.array(res)
    return np.array([np.interp(newTs, ts, signal.convolve(sig, wav(N, i), mode="same")) for i in wid])

def highpass_filter(y, sr, stp, bnd):
    """Applies a high pass filter using convolution.
    
    Args:
        y (array): The signal to apply the filter to.
        sr (float): The sampling rate of the signal.
        stp (float): The frequency that the cutoff should be.
        bnd (float): The band width after the cutoff which is still somewhat suppressed.
    
    Returns:
        array: Filtered signal.
    """
    fc = stp/sr
    b = bnd/sr
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1
    n = np.arange(N)
     
    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = np.blackman(N)
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)
    
    # reverse function
    sinc_func = -sinc_func
    sinc_func[int((N - 1) / 2)] += 1
    
    return np.convolve(y, sinc_func)

def mod2(X):
    """Gets the absolute magnitude squared of X.
    
    Args:
        X (array): X axis values.
        
    Returns:
        array: Magnitude of X squared.
    """
    return abs(X)**2

def reduceSize(x, sz):
    """Reduces size to below sz, while keeping as many elements as possible (at most len(x)%sz-1 elements will be lost).
    Probably more useful to use the interp function from numpy.
    
    Args:
        x (array): Array to reduce size of.
        sz (int): New array size to aim for.
    
    Returns:
        array: Reduced array.
    """
    
    N = len(x)
    k = int(N/np.ceil(N/sz))
    return (x[:-(N%k)] if N%k!=0 else x).reshape(-1, k, order='F').mean(axis=0)

def cinitFunc(x):
    """Initial wave function before imaginary time propagation.
    
    Args:
        x (array): X axis values.
    
    Returns:
        int: 1, can be changed later to better match initial conditions.
    """
    return np.ones(x.shape)

def tukeyFlatWindow(n, left, alpha=0.5):
    """Produces a window with a standard tukey taper on one side and no taper on the other.

    Args:
        n (int): Number of points for window.
        left (bool): Whether the taper should be on the left (true) or the right (false).
        alpha (float): Amount of window within tapered region. Defaults to 0.5.

    Returns:
        array: Window values.
    """

    vals = signal.tukey(n, alpha)
    if left:
        vals[int(n/2):] = 1
    else:
        vals[:int(n/2)] = 1

    return vals
    
def getFFT(y, dx, win=0):
    """Gets FFT of y, only positive frequencies.
    
    Args:
        y (array): Signal to take an fft of.
        dx (float): Sampling period of data.
        win (array): Window to utilize. If 0, Hanning window is used. Defaults to 0.
    
    Returns:
        array: Frequencies associated with FFT.
        array: Absolute value squared of FFT.
    """
    if win is int:
        fourier = scipy.fftpack.fft(y*np.hanning(len(y)))
    else:
        fourier = scipy.fftpack.fft(y*win)
    freq = scipy.fftpack.fftfreq(fourier.size, dx)
    return freq, fourier

def getFFTPowerSpectrum(y, dx):
    """Gets FFT Power Spectrum, only positive frequencies.
    
    Args:
        y (array): Signal to take an fft of.
        dx (float): Sampling period of data.
    
    Returns:
        array: Frequencies associated with FFT.
        array: Absolute value squared of FFT.
    """
    #fourier = scipy.fftpack.fft(y*np.hanning(len(y)))
    fourier = scipy.fftpack.fft(y*signal.tukey(len(y)))
    freq = scipy.fftpack.fftfreq(fourier.size, dx)
    fourier = abs(fourier)**2
    return freq[:int(len(freq)/2)], fourier[:int(len(freq)/2)]
#    return freq, fourier

def getFFTPowerSpectrumWindowed(y, dx, winPos, winStd):
    """Gets FFT Power Spectrum, only positive frequencies. The signal is windowed with a gaussian.
    
    Args:
        y (array): Signal to take an fft of.
        dx (float): Sampling period of data.
        winPos(float): Position of maximum of window.
    
    
    Returns:
        array: Frequencies associated with FFT.
        array: Absolute value squared of FFT.
    """
    gaus = np.exp(-((np.array(range(len(y)))-winPos)/winStd)**2/2)
    fourier = scipy.fftpack.fft(y*gaus)
    freq = scipy.fftpack.fftfreq(fourier.size, dx)
    fourier = np.real(fourier*np.conj(fourier))
    return freq[:int(len(freq)/2)], fourier[:int(len(freq)/2)]

def getEnergySpecFromCurrent(cur, psi, nbin, binmin, binmax):
    """Gets electron energy spectrum from probability current and the wave function at that point.
    
    Args:
        cur (array): Probability current.
        psi (array): Wave function for each current measurement.
        nbin (int): Number of bins.
        binmin(float): Minimum bin energy value, eV.
        binmax(float): Maximum bin energy value, eV.
    
    
    Returns:
        array: Energies of bins.
        array: Count in each bin.
    """

    E = (me*(cur/(abs(psi)**2))**2/2)/eV_J
    h,_ = np.histogram(E, nbin, (binmin, binmax))
    return  np.linspace(binmin, binmax, nbin), h
    
def getIdealDX(Emax, Xmax, allowedAccError):
    """Gets ideal dx value based on maximum electric field and maximum x value.
    
    Args:
        Emax (float): Maximum electric field, GV/m.
        Xmax (float): Maximum distance electron may travel, m.
        allowedAccError (float): Maximum error allowed in acceleration, decimal.
    
    Returns:
        float: Ideal dx for given settings.
    """
    return np.sqrt(allowedAccError/(1.25e19*Emax*Xmax)) if Emax != 0 else 0.001*Xmax

def getIdealMaxX(Emax, lam):
    """Gets maximum position of a classical electron if it tunnels into the field just before the potential begins to lower.
    
    Args:
        Emax (float): Maximum electron field, V/m.
        lam (float): Wavelength of light field, m.
    
    Returns:
        float: Maximum travel distance of classical electron in light field, m.
    """
    return Emax*lam**2*qe/(2*c**2*me*np.pi)

def getTravelDist(maxE, t):
    """Gets maximum distance an electron of energy maxE may travel in time t.
    
    Args:
        maxE (float): Maximum electron energy, J.
        t (float): Maximum time of flight, s.

    Returns:
        float: Maximum travel distance of classical electron of specified energy, m.
    """
    return t*np.sqrt(2*maxE/me)

def getIdealMaxT(Xmax, measEMin):
    """Gets minimum time required for an electron of specified energy to traverse the simulation.
    
    Args:
        Xmax (float): Maximum distance an electron must travel in simulation, m.
        measEMin (float): Minimum energy electron that needs to make it to the edge, J.
    
    Returns:
        float: Minimum extra time after pulse required for electrons of energy measEMin to reach Xmax, s.
    """
    return Xmax*np.sqrt(me/(2*measEMin))

def maxTimeStep(Vmax, dx):
    """Gets maximum time step that is stable for the euler method.
    
    Args:
        Vmax (float): Maximum potential energy difference in simulation, J.
        dx (float): Spatial step size in simulation, m.
    
    Returns:
        float: Maximum allowable dt, s.
    """
    return hbar / ((hbar**2 / (me * dx**2)) + 0.5*Vmax)

def getKeldysh(W, E, lam):
    """Gets the Keldysh parameter for given settings.
    
    Args:
        W (float): Work function of metal, J.
        E (float): Maximum electric field of light field, V/m.
        lam (float): Wavelength of light field, m.
    
    Returns:
        float: Keldysh parameter.
    """
    return np.sqrt(W/(2*getPond(E, lam)))

#calculates ponderomotive energy
def getPond(E, lam):
    """Calculates ponderomotive energy for given settings.
    
    Args:
        E (float): Maximum electric field of light field, V/m.
        lam (float): Wavelength of light field, m.
    
    Returns:
        float: Ponderomotive energy, J.
    """
    w = np.pi*2*c/lam
    Up = 0.25*(E*qe)**2/(w**2*me)
    return Up

class Potential(object):
    """Arbitrary potential object.
    
    Args:
        X (array): X axis values.
    """
    def __init__(self, X):
        self.V = np.zeros(X.shape)
        self.name = "none"
    
    def getMaxV(self):
        """Gets maximum voltage.
        
        Returns:
            double: Maximum voltage, V.
        """
        return max(self.V)-min(self.V)
        
    def getV(self):
        """Gets voltage.
        
        Returns:
            array: Voltage values, V.
        """
        return self.V
    
class ShieldedPotential(Potential):
    """Shielded atomic potential.
    
    Args:
        X (array): X axis values.
        x0 (float): Position of atomic layer, m.
        d (float): Atomic spacing in lattice, m.
        Z (float): Number of protons in the shielded potential.
        lam (float): Decay distance of shielding, m.
    """
    def __init__(self, X, x0, d, Z, lam):
        self.V = -Z*(qe**2)/(2*e0*(d**2)/lam)*np.exp(-np.abs(X-x0)/lam)
        self.name = "Shielded"
        
class ShieldedLattice(Potential):
    """Shielded atomic potential lattice.
    
    Args:
        X (array): X axis values.
        x0 (float): Position of atomic layer, m.
        d (float): Atomic spacing in lattice, m.
        Z (float): Number of protons in the shielded potential.
        lam (float): Decay distance of shielding, m.
        n (int, optional): Number of layers included in potential.
    """
    def __init__(self, X, x0, d, Z, lam, n=1):
        self.V = ShieldedPotential(X, x0, d, Z, lam).getV()
        for i in range(1, n):
            self.V += ShieldedPotential(X, x0-d*i, d, Z, lam).getV()
        self.name = "%d Shielded"%n
        
class UnshieldedPotential(Potential):
    """Unshielded atomic potential.
    
    Args:
        X (array): X axis values.
        x0 (float): Position of atomic layer, m.
        d (float): Atomic spacing in lattice, m.
        Z (double): Number of protons in the unshielded potential.
    """
    def __init__(self, X, x0, d, Z):
        self.V = -Z*(qe**2)/(2*e0*(d**2))*(np.sqrt(d**2/np.pi+(X-x0)**2)-np.abs(X-x0))
        self.name = "Unshielded"

class WachterJelliumPotential(Potential):
    """Wachter's jellium potential.
    
    Args:
        X (array): X axis values.
        Ef (float): Fermi energy, J.
        W (float): Work function, J.
    """
    def __init__(self, X, Ef, W):
        X = X/a0
        Ef = Ef/auE
        W = W/auE
        V0 = Ef+W
        nbulk = 1/(3*(np.pi**2))*(2*Ef)**2
        rs = np.cbrt(3/(4*np.pi*nbulk))
        zim = -0.2*rs + 1.25
        kf = np.sqrt(2*Ef)
        b = kf
        A = 4*V0/b-1
        B = V0/(4*V0/b-1)
        
        def getVals(i):
            if i < zim:
                return -V0/(A*np.exp(B*(i))+1)
            else:
                return -(1-np.exp(-b*(i)))/(4*(i))*(-V0/(A*np.exp(B*(zim))+1))/(-(1-np.exp(-b*(zim)))/(4*(zim))) #Includes correction for Wachter's discontinuous jellium potential, forcing continuity.
        
        self.V = np.array([getVals(i) for i in X])
        self.V *= auE
        
        self.zim = zim*a0
        
        self.name = "Jellium"
        
class WachterAtomPotential(Potential):
    """Wachter's soft-core atomic potential.
    
    Args:
        X (array): X axis values.
        Ef (float): Fermi energy, J.
        x0 (float, optional): Position of atomic layer, m.
    """
    def __init__(self, X, Ef, x0=0):
        Ef = Ef/auE
        nbulk = 1/(3*np.pi**2)*(2*Ef)**2
        kft = 6*np.pi*nbulk/Ef
        lamft = 2*np.pi/kft
        X = X/a0
        
        self.V = -1/(1+abs(X-x0))*np.exp(-abs(X-x0)/lamft)*auE
        
        self.name = "Wachter Atom"
        
class WachterComposite(Potential):
    """Combination of Wachter's jellium potential and soft-core potential.
    
    Args:
        X (array): X axis values.
        Ef (float): Fermi energy, J.
        W (float): Work function, J
    """
    def __init__(self, X, Ef, W):
        self.V = WachterJelliumPotential(X, Ef, W).getV() + WachterAtomPotential(X, Ef).getV()
        self.name = "Wachter Combined"

class SquareWell(Potential):
    """Finite square well potential.
    
    Args:
        X (array): X axis values.
        Vwell (float): Depth of well, J.
        L (float): Width of well, m.
    """
    def __init__(self, X, Vwell, L):
        def getVal(i):
            if i > L or i < 0:
                return 0
            else:
                return 1
        self.V = np.array([getVal(i) for i in X])*(-Vwell)
        self.name = "Square Well"
    

class CompositePotentialFunction(Potential):
    """Combination of multiple potential functions.
    
    Args:
        X (array): X axis values.
        funcs (list): Potential functions to combine.
    """
    def __init__(self, X, funcs):
        self.V = np.zeros(X.shape)
        for i in funcs:
            self.V += i.getV()
        
        self.name = "Composite: "
        for i in funcs:
            self.name += i.name+" "


class Envelope(object):
    """Envelope for light pulse (default constant)."""
    def __init__(self):
        self.name = "Constant"
    
    def getMax(self, t):
        """Gets envelope value at current time.
        
        Args:
            t (float): Time to evaluate envelope.
        
        Returns:
            float: Envelope value.
        """
        return 1.0

class Pulse(Envelope):
    """Gaussian light pulse.
    
    Args:
        tau (float): Time constant for gaussian (sigma), s.
        tmax (float): Center of gaussian pulse, s.
    """
    def __init__(self, tau, tmax):
        self.name = "Pulse"
        self.tau = tau
        self.tmax = tmax
        self.tmul = self.tmax/self.tau/5
    
    def getMax(self, t):
        """Gets envelope value at current time.
        
        Args:
            t (float): Time to evaluate envelope.
        
        Returns:
            float: Envelope value.
        """
        if t > self.tau/self.tmul:
            return np.exp(-(t-self.tmax)**2/(2*self.tau**2))
        else:
            return np.exp(-(t-self.tmax)**2/(2*self.tau**2))*abs(3*(t/self.tau*self.tmul)**2-2*(t/self.tau*self.tmul)**3)


class FlatTopPulse(Envelope):
    """Gaussian light pulse, with a flat top.

    Args:
        tau (float): Time constant for gaussian (sigma), s.
        tmax (float): Center of gaussian pulse, s.
        tflat (float): Length of flat top.
    """

    def __init__(self, tau, tmax, tflat):
        self.name = "Pulse"
        self.tau = tau
        self.tmax = tmax
        self.tflat = tflat
        self.tmul = self.tmax / self.tau / 5
        self.leftmax = tmax-tflat/2
        self.rightmax = tmax+tflat/2

    def getMax(self, t):
        """Gets envelope value at current time.

        Args:
            t (float): Time to evaluate envelope.

        Returns:
            float: Envelope value.
        """
        if t < self.tau / self.tmul:
            return np.exp(-(t - self.leftmax) ** 2 / (2 * self.tau ** 2)) * abs(
                3 * (t / self.tau * self.tmul) ** 2 - 2 * (t / self.tau * self.tmul) ** 3)
        elif t < self.leftmax:
            return np.exp(-(t - self.leftmax) ** 2 / (2 * self.tau ** 2))
        elif t < self.rightmax:
            return 1
        else:
            return np.exp(-(t - self.rightmax) ** 2 / (2 * self.tau ** 2))


class TimeDependentPotential(Potential):
    """Arbitrary potential function that depends on time.
    
    Args:
        X (array): X axis values.
    """
    def __init__(self, X):
        self.V = np.zeros(X.shape)
        self.name = "tdnone"
    
    def getV(self, t):
        """Gets potential function at current time.
        
        Args:
            t (float): Time to evaluate potential.
        
        Returns:
            array: Potential at time t.
        """
        return self.V
    
class LightField(TimeDependentPotential):
    """Light field potential.
    
    Args:
        X (array): X axis values.
        Emax (float): Maximum electric field, V/m.
        lam (float): Wavelength of light, m.
        phase (float): Phase of light (3pi/2 is sine-like with maximum decrease before maximum increase), rad.
        env (Envelope): Envelope for light pulse.
        minX (float): Minimum X-value that light field will exist in, m.
        maxX (float): Maximum X-value that light field will exist in, m.
        tmax (float): Maximum time of simulation, s.
    """
    def __init__(self, X, Emax, lam, phase, env, minX, maxX, tmax):
        self.X = X
        self.V = np.zeros(X.shape)
        self.name = "Light Field, " + env.name + " Envelope"
        self.Emax = Emax
        self.lam = lam
        self.f = c/lam
        self.env = env
        self.maxX = maxX
        self.minX = minX
        self.phase = phase
        self.tmax = tmax
        
        
        def getMask(i):
            if i <= maxX and i >= minX:
                return i-maxX #i-minX
            elif i > maxX:
                return 0.0 #maxX-minX
            else:
                return minX-maxX #0
        self.mask = np.array([getMask(i) for i in X])
        
        def getOutside(i):
            if i > (minX*0.95+maxX*0.05):
                return 1
            else:
                return 0
        self.outsideMask = np.array([getOutside(i) for i in X])
    
    def getV(self, t):
        """Gets potential function at current time.
        
        Args:
            t (float): Time to evaluate potential.
        
        Returns:
            array: Potential at time t.
        """
        if t != 0:
            return self.mask*self.env.getMax(t)*self.Emax*qe*np.cos(self.getPhase(t))
        else:
            return 0
    
    def getPhase(self, t):
        """Gets current phase of light field.
        
        Args:
            t (float): Time to get phase, s.
        
        Returns:
            float: Phase, not normalized to any limits, rad.
        """
        return self.phase + np.pi*2*self.f*(t-self.tmax)
    
    def getMaxV(self):
        """Gets maximum potential energy for electron in light field (if it were constant).
        
        Returns:
            float: Maximum potential energy from light field, J.
        """
        return abs(self.Emax*(self.maxX-self.minX)*qe)
        
class ConstantField(TimeDependentPotential):
    """Non-oscillating electric field.
    
    Args:
        X (array): X axis values.
        Emax (float): Electric field strength, V/m.
        env (Envelope): Envelope for electric field.
        minX (float): Minimum X value for field to apply to, m.
        maxX (float): Maximum X value for field to apply to, m.
    """
    def __init__(self, X, Emax, env, minX, maxX):
        self.X = X
        self.V = np.zeros(X.shape)
        self.name = "Constant Field, " + env.name + " Envelope"
        self.Emax = Emax
        self.env = env
        self.maxX = maxX
        self.minX = minX
        def getOutside(i):
            if i > (minX*0.95+maxX*0.05):
                return 1
            else:
                return 0
        self.outsideMask = np.array([getOutside(i) for i in X])
        
    def getV(self, t):
        """Gets potential function at current time.
        
        Args:
            t (float): Time to evaluate potential.
        
        Returns:
            array: Potential at time t.
        """
        if t != 0:
            return -self.X*self.Emax*qe
        else:
            return self.X*0
            
    def getPhase(self, t):
        """Gets current phase of light field.
        
        Args:
            t (float): Time to get phase, s.
        
        Returns:
            float: Phase, not normalized to any limits, rad.
        """
        return 0
        
    def getMaxV(self):
        """Gets maximum potential energy for electron in light field (if it were constant).
        
        Returns:
            float: Maximum potential energy from light field, J.
        """
        return self.Emax*(self.maxX-self.minX)*qe

class PotentialFunction(TimeDependentPotential):
    """Combined static potential and time dependent potential, for combining a metallic potential and a light field.
    
    Args:
        X (array): X axis values.
        initPot (Potential): Initial potential function (metal).
        lightFunc (TimeDependentPotential): Changing potential function (light field).
        keepInitialPotential (bool, optional): Removes initial potential for non-zero t.
    """
    def __init__(self, X, initPot, lightFunc, keepInitialPotential=True):
        self.X = X
        self.iV = initPot.getV()
        self.lightFunc = lightFunc
        self.type = initPot.name + " and " + lightFunc.name
        self.keepInitialPotential = keepInitialPotential
        
    def getV(self, t):
        """Gets potential function at current time.
        
        Args:
            t (float): Time to evaluate potential.
        
        Returns:
            array: Potential at time t.
        """
        if (not self.keepInitialPotential) and t != 0:
            return self.X*0 + self.lightFunc.getV(t)
        else:
            return self.iV + self.lightFunc.getV(t)
    
    def getMaxV(self):
        """Gets maximum potential energy for electron in light field (if it were constant).
        
        Returns:
            float: Maximum potential energy from light field, J.
        """
        return max(self.iV)-min(self.iV) + self.lightFunc.getMaxV()



class WaveFunction(object):
    """Wave function object, used for performing the simulation.
    
    Args:
        pot (PotentialFunction): Potential function to simulate wave function in.
        decayBufferSize (float): Size of absorptive edge, m.
        bufferLeft (bool): Whether to place an absorptive edge on the left part of the simulation (usually in metal).
        tprec (float, optional): Additional precision in t beyond euler method's maximum dt (new dt = old dt/tprec). Defaults to 1.
    """
    def __init__(self, pot, decayBufferSize, bufferLeft, tprec=1):
        self.V = pot
        self.iV = pot.getV(0)
        self.curV = self.iV
        self.X = pot.X #save X-data to wave func
        self.its = 0 #initialize number of imaginary step iterations
        self.dx = self.X[1] - self.X[0] #Extract dx from the given range
        self.dt = 0.5 * maxTimeStep(pot.getMaxV(), self.dx)/tprec #determine an ideal dt for the simulation (maximum for euler method)
        self.N = len(self.X) #number of data points
        
        self.psi = np.array(cinitFunc(self.X), dtype = np.complex_) #initialize wave function, make complex
        
        self.c1 = hbar * self.dt / (2 * me * self.dx**2) #constant for solving TDSE (Euler)
        self.c2 = self.dt / hbar #other constant
        
        idx = np.arange(self.N) #create array of indices corresponding to each position
        self.lm1 = idx.take(range(-1, self.N-1), mode='wrap') #shift the indices to the left
        self.lp1 = idx.take(range(1, self.N+1), mode='wrap') #shift indices to right
        self.lp1[self.N-1] = self.N-1 #forces edge of simulation to act like a "free ended string:
        self.lm1[0] = 0               #instead of being transmitted to other side of simulation
        
        self.icurT = 0 #set imaginary curtime to 0
        self.rcurT = 0 #set real curtime to 0
        self.grounded = False #not grounded yet
        self.normalize() #normalize the wave function
        
        self.alpha=hbar/(2*me)*self.dt/(2*(self.dx**2))*1j #similar to c1 and c2, for CN method
        self.beta=self.dt/(2*hbar)*1j #ditto

        #get matrix required for using solve_banded
        A = np.zeros((3,self.N), dtype=complex)
        A[0,:] = np.ones(self.N)*(-self.alpha)
        A[1,:] = 1+2*self.alpha
        A[2,:] = A[0,:]
        self.A0 = A
        
        A = np.zeros((3,self.N), dtype=complex)
        A[1,:] = 1
        self.Anm = A
        
        A = np.zeros((3,self.N), dtype=complex)
        A[0,:] = np.ones(self.N)*(-self.alpha*-1j)
        A[1,:] = 1+2*self.alpha*-1j
        A[2,:] = A[0,:]
        self.A0im = A
        del A
        
        maxx = self.X[-1] #get the maximum x-val
        minx = self.X[0] #get min x-val
        
        #generates a mask which makes wave functions at the edge decay
        def getDecayMaskVal(i):
            if i >= maxx-decayBufferSize:
                return (abs(i-maxx+decayBufferSize)/decayBufferSize)**2
            elif bufferLeft and i <= minx+decayBufferSize:
                return (abs(i-minx-decayBufferSize)/decayBufferSize)**2
            else:
                return 0
        self.decayEdgeMask = [getDecayMaskVal(i) for i in self.X] #create the mask
    
    def normalize(self):
        """Normalizes the wave function."""
        self.psi /= np.sqrt(spin.trapz(self.prob(), dx=self.dx))
    
    def solveTridiagonal(self, odval, ids, d):
        """Solves a tridiagonal matrix with same secondary diagonals (no variation). NOT WORKING, EXTREMELY INEFFICIENT BECAUSE PYTHON"""
        n = len(d)
        
        cp = np.empty(n-1, dtype=np.complex_)
        lval = complex(0)
        for i in range(n-1):
            lval = odval/(ids[i]-odval*lval)
            cp[i] = lval
            
        dp = np.empty(n, dtype=np.complex_)
        lval = d[0]/ids[0]
        dp[0] = lval
        for i in range(1, n):
            lval = (d[i]-odval*lval)/(ids[i]-odval*cp[i-1])
            dp[i] = lval
            
        xs = np.empty(n, dtype=np.complex_)
        lval = complex(0)
        for i in range(n-1, -1, -1):
            lval = dp[i]-odval*lval
            xs[i] = lval
        
        return xs
    
    def stepCN(self, changeTime=True, imaginary=False):
        """Performs a step using the Crank-Nicolson method (preferred step method).
        
        Args:
            changeTime (bool, optional): Increments rcurT by dt if true. Defaults to True.
            imaginary (bool, optional): Performs imaginary time propagation if true. Defaults to False.
        """
        Vo = self.V.getV(self.rcurT)
        self.curV = Vo
        if imaginary:
            Vn = Vo
            al = self.alpha*-1j
            be = self.beta*-1j
            A = self.A0im + self.Anm*be*Vn
        else:
            Vn = self.V.getV(self.rcurT+self.dt)
            al = self.alpha
            be = self.beta
            A = self.A0+self.Anm*be*Vn

        self.psi = sp.linalg.solve_banded((1,1), A, 
                                          self.psi*(1-2*al-be*Vo)+(self.psi[self.lm1]+self.psi[self.lp1])*al,
                                          overwrite_ab=True, overwrite_b=True, check_finite=False)
#        self.psi = self.solveTridiagonal(-al, Vn+1+2*al, self.psi*(1-2*al-be*Vo)+(self.psi[self.lm1]+self.psi[self.lp1])*al)

        if imaginary:
            self.normalize()
            self.icurT += self.dt
        elif changeTime:
            self.rcurT += self.dt
    
    def step(self, changeTime=True):
        """Performs a step using the Euler method (unused).
        
        Args:
            changeTime (bool, optional): Increments rcurT by dt if true. Defaults to True.
        """
        V = self.V.getV(self.rcurT)
        self.curV = V
        #Euler method, repeated prec times
        psi = self.psi
        self.psi.imag = psi.imag + (self.c1*(psi.real[self.lp1] - 2.0*psi.real + psi.real[self.lm1]) - (self.c2*V*psi.real))
        self.psi.real = psi.real - (self.c1*(psi.imag[self.lp1] - 2.0*psi.imag + psi.imag[self.lm1]) - (self.c2*V*psi.imag))
#            self.psi = psi + np.complex(0.0+1.0j)*(self.c1*(psi[self.lp1]-2.0*psi+psi[self.lm1])-self.c2*V*psi)/float(prec)
#            self.psi[0] = 0 #acts as an infinite potential barrier... an actual large potential should be in the simulation for best, most physical results (this take on 
                            #infinite potential does not conserve probability for finite dt)
        if changeTime:
            self.rcurT += self.dt
        #self.normalize()
    
    #performs an imaginary time step (Euler), then renormalizes, UNUSED
    def stepImaginary(self):
        """Performs an imaginary time step using Euler method (unused)."""
        V = self.iV
        psi = self.psi
        self.psi.imag = psi.imag + self.c1 * (
            psi.imag[self.lp1] - 2*psi.imag + psi.imag[self.lm1]) - (
            self.c2 * V * psi.imag)
        self.psi.real = psi.real + self.c1 * (
            psi.real[self.lp1] - 2*psi.real + psi.real[self.lm1]) - (
            self.c2 * V * psi.real)
#        self.psi[0] = 0
        
        self.icurT += self.dt
        self.normalize()
        self.its += 1
    
    def decayEdge(self):
        """Decays the edge of the simulation, such that any waves at the edge are removed as though they escaped the system (similar to imaginary potential). CONSIDER PLACING IMAGINARY POTENTIAL FOR REGULAR stepCN"""
        self.psi += self.c1*(self.psi[self.lp1]-2*self.psi+self.psi[self.lm1])*self.decayEdgeMask
        
    def plotWaveFuncReal(self):
        """Plots the wave function. UNUSED"""
        return plt.plot(self.X, self.psi.real)
    
    def getEnergy(self):
        """Gets the total energy of the electron.
        
        Returns:
            float: Total energy of electron, J.
        """
        return spin.trapz((self.psi.conj()*self.hamOp(self.psi)).real, dx=self.dx)
    
    def getFFT(self):
        """Performs an FFT on the wave function. UNUSED"""
        fourier = np.fft.fft(self.psi.real)
        freq = np.fft.fftfreq(fourier.size, self.dx)
        fourier = np.log(np.real(fourier*np.conj(fourier)))
        return [x for (x, y) in sorted(zip(freq, fourier))], [y for (x, y) in sorted(zip(freq, fourier))]
        
    def probCurrent(self):
        """Gets the probability current of the wave function.
        
        Returns:
            array: Probability current spectrum.
        """
        d = self.psi[self.lp1]-self.psi[self.lm1]
        return -hbar/(4*me*self.dx)*(d.conj()*self.psi - self.psi.conj()*d).imag
    
    def probCurrentPt(self, i):
        """Gets the probability current at a single point.
        
        Args:
            i (int): Index to get probability current at.
        
        Returns:
            float: Probablity current at i.
        """
        d = self.psi[i+1]-self.psi[i-1]
        return -hbar/(4*me*self.dx)*(np.conj(d)*self.psi[i]-np.conj(self.psi[i])*d).imag
    
    def wavePhaseDerPt(self, i):
        """Gets the phase gradient at a single point.
        
        Args:
            i (int): Index to get phase gradient at.
            
        Returns:
            float: Phase gradient at i.
        """
        psi = self.psi
        return ((psi[i+1]-psi[i-1])/psi[i]).imag/(2*self.dx)
    
    def prob(self):
        """Gets the probability density spectrum for the wave function.
        
        Returns:
            array: Probability density.
        """
        #return (self.psi*np.conj(self.psi)).real
        return abs(self.psi)**2
    
    def totalProb(self):
        """Gets the total probability in the system.
        
        Returns:
            float: Total probability.
        """
        return np.sqrt(spin.trapz(abs(self.psi)**2, dx=self.dx))
    
    def expectXField(self):
        """Gets the expectation value of position x, taking into account only the wave function in the light field.
        
        Returns:
            float: Expectation value of x in light field, m.
        """
        npsi = self.psi*(self.V.lightFunc.outsideMask)
        return spin.trapz(abs(npsi)**2*self.X)/spin.trapz(abs(npsi)**2)
      
    def expectPField(self):
        """Gets the expectation value of momentum p, taking into account only the wave function in the light field.
        
        Returns:
            float: Expectation value of p in light field, kgm/s
        """
        npsi = self.psi*(self.V.lightFunc.outsideMask)
        return hbar*spin.trapz((npsi*(npsi[self.lm1]-npsi[self.lp1]).conj()).imag)/(2.0*self.dx*spin.trapz(abs(npsi)**2))

    def expectX(self):
        """Gets the expectation value of x.
        
        Returns:
            float: Expectation value of x, m.
        """
        return spin.trapz(abs(self.psi)**2*self.X, dx = self.dx)
    
    def expectX2(self):
        """Gets the expectation value of x^2.
        
        Returns:
            float: Expectation value of x^2, m.
        """
        return spin.trapz(abs(self.psi)**2*self.X**2, dx = self.dx)
    
    def sigX(self):
        """Gets the uncertainty in x.
        
        Returns:
            float: Uncertainty in x, m.
        """
        return np.sqrt(self.expectX2()-self.expectX()**2)
    
    def expectP(self):
        """Gets the expectation value of p.
        
        Returns:
            float: Expectation value of p, kgm/s.
        """
        return hbar*spin.trapz((self.psi.conj()*(self.psi[self.lm1]-self.psi[self.lp1])).imag)/2.0

    def spectrumP(self):
        """Applies the momentum operator to the wave function.
        
        Returns:
            array: Momentum operated wave function.
        """
        return (hbar/(2.0*self.dx)*-1j)*(self.psi[self.lm1]-self.psi[self.lp1])

    #gets expectation acceleration
    def expectA(self):
        """Gets the expectation value of acceleration, a.
        
        Returns:
            array: Expectation value of a, N/kg.
        """
        return spin.trapz((abs(self.psi)**2*(self.curV[self.lm1]-self.curV[self.lp1])).real, dx=self.dx)/(2*me*self.dx)
    
    #gets expectation acceleraton^2
    def expectA2(self):
        """Gets the expectation value of a^2.
        
        Returns:
            array: Expectation value of a^2, N^2/kg^2.
        """
        return spin.trapz((abs(self.psi)*(self.curV[self.lm1]-self.curV[self.lp1])).real**2, dx=self.dx)/(4*me**2*self.dx)
    
    def findGroundState(self, its):
        """Uses imaginary time propagation through stepCN to get the ground state.
        
        Args:
            its (int): Number of imaginary steps to make.
        
        Returns:
            float: Difference in between energies before and after imaginary time propagation, absolute value, J.
        """
        iE = self.getEnergy()
        for i in range(its):
            self.stepCN(imaginary=True)
        return abs(iE-self.getEnergy())

    def steadyStateSolve(self, E):
        hp = int(len(self.psi)/2)
        self.psi[hp] = 1
        psip = 0
        for i in range(hp, 0, -1):
            psipp = -2*me/hbar**2*(self.iV[i]-E)*self.psi[i]
            psip = -psipp*self.dx+psip
            self.psi[i-1] = -psip*self.dx + self.psi[i] - 0.5*psipp*self.dx**2
        psip = 0
        for i in range(hp, len(self.psi)-1):
            psipp = -2*me/hbar**2*(self.iV[i]-E)*self.psi[i]
            psip = psipp*self.dx+psip
            self.psi[i+1] = psip*self.dx + self.psi[i] + 0.5*psipp*self.dx**2
        self.normalize()
        
    
    def hamOp(self, wav):
        """Applies the hamiltonian operator to a function.
        
        Args:
            wav (array): Function to apply hamiltonian operator to. Must be same size as this wave function.
            
        Returns:
            array: Hamiltonian operated function.
        """
        return -hbar**2/(2*me*self.dx**2)*(wav[self.lp1]-2*wav+wav[self.lm1])+self.curV*wav
    
    def spectrumA(self):
        """Applies the acceleration operator to the wave function.
        
        Returns:
            array: Acceleration operated wave function.
        """
        return (self.curV[self.lm1]-self.curV[self.lp1])*self.psi/(2*me*self.dx)
    
    def aOp(self, wav):
        """Applies the acceleration operator to a given function.
        
        Args:
            wav (array): Function to apply acceleration operator to. Must be same size as this wave function.
            
        Returns:
            array: Acceleration operated function.
        """
        return (self.curV[self.lm1]-self.curV[self.lp1])*wav/(2*me*self.dx)
        
    
def generateData(X, Emax, lam, bufferSize, tau=8e-15, totTime=100e-15, centMul=8, detPos=0, bufferLeft=True, tprec=1, phase=0, typ=2, getGroundState=True, initialState=0, plotSaves=True, itsBetweenSave=250, stopWhenFinished=True, saveInMiddle=False, savePsi=True, savePsiRes=100, saveWhenFinished=False, printProgress=False, subFolder='default', fname="%s"%time.strftime("/%y%m%d%H%M%S")):
    """Helper function used to generate data or to animate the simulation of HHG in 1-D.
    
    Args:
        X (array): X axis values.
        Emax (float): Maximum electric field, V/m.
        lam (float): Wavelength of light field, m.
        bufferSize (float): Size of absorptive edge, m.
        tau (float, optional): Standard deviation of gaussian pulse, s. Defaults to 5 fs.
        centMul (float, optional): Time from the start of the simulation to the peak of the gaussian pulse, in units of tau. Larger delays correspond to less leaking of the wave function due to the pulse starting (less noise). Defaults to 8.
        detPos (float, optional): Position of virtual detector and edge of light field, m. Defaults to 0 (90% from edge).
        bufferLeft (bool, optional): Whether to place an absorptive edge on the left. Defaults to True.
        tprec (float, optional): Precision in t. Defaults to 1.
        phase (float, optional): Carrier envelope phase of light field. Defaults to 0 (cosine like, preferred sine-like is 3pi/2).
        typ (int, optional): Selects the initial potential function...
            0 = Wachter Jellium and Soft-Core Atomic
            1 = Wachter Jellium and Unshielded
            2 = Wachter Jellium and Shielded (Default)
            3 = Wachter Jellium and soft Shielded
            4 = Wachter Jellium and a combination of Shielded and Unshielded
            5 = Wachter Jellium and a metal 1 layers deep of Shielded
            6 = Wachter Jellium and a metal 5 layers deep of Shielded
            7 = 0 Potential
        getGroundState (bool, optional): Whether or not to obtain the ground state. Defaults to True.
        initialState (array, optional): Initial state of the wave function. Defaults to 0 (find ground state).
        plotSaves (bool, optional): Whether or not to plot a set of summary plots while simulating. Defaults to True.
        itsBetweenSave (int, optional): Number of iterations of simulation between plotting (or saving if saveInMiddle). Defaults to 250.
        stopWhenFinished (bool, optional): Whether or not to terminate the simulation when the maximum time has been reached. Defaults to True.
        saveInMiddle (bool, optional): Whether or not to save mid simulation (not recommended, very slow). Defaults to False.
        savePsi (bool, optional): Whether or not to save the wave function periodically for plotting the probability distribution. Defaults to True.
        savePsiRes (int, optional): Resolution of the savePsi feature. Defaults to 100.
        saveWhenFinished (bool, optional): Whether or not to save the results when the simulation is finished. Defaults to True.
        printProgress (bool, optional): Whether or not to print the progress (finding ground state and simulation). Defaults to False.
        subFolder (str, optional): Name of subfolder to save files to. Defaults to 'default'.
        fname (str, optional): Name of file to save data as. Defaults to a name based on the time of running this method.
    """
    if saveWhenFinished:
        fold = "data/" + subFolder #create folders for data
        fil = fold+"/"+fname
        if not os.path.exists(fold):
            os.makedirs(fold)
    
    cent = centMul*tau
    
    env = Pulse(tau, cent)
    # env = FlatTopPulse(tau, cent, tau*2)
    #env = Envelope()
    #ipot = ShieldedPotential(X, 0, 2.5e-10, 1.74, 2.5e-10)
    #ipot = WachterJelliumPotential(X, 7*eV_J, 4.7*eV_J)
    if typ == 0:
        ipot = WachterComposite(X, 9.2*eV_J, 6.2*eV_J)
    elif typ == 1:
        ipot = CompositePotentialFunction(X, [WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J), UnshieldedPotential(X, 0, 2.5e-10, 1.3)])
    elif typ == 2:
        ipot = CompositePotentialFunction(X, [WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J), ShieldedPotential(X, 0, 2.5e-10, 1.74, 1e-10)])
    elif typ == 3:
        ipot = CompositePotentialFunction(X, [WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J), ShieldedPotential(X, 0, 2.5e-10, 0.7, 2.5e-10)])
    elif typ == 4:
        ipot = CompositePotentialFunction(X, [WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J), ShieldedPotential(X, 0, 2.5e-10, 72, 1e-10), UnshieldedPotential(X, 0, 2.5e-10, 2)])
    elif typ == 5:
        ipot = CompositePotentialFunction(X, [WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J), ShieldedLattice(X, -2.5e-10, 2.5e-10, 1.74*0.5, 1e-10, n=1), ShieldedPotential(X, 0, 2.5e-10, 1.74, 1e-10)])
    elif typ == 6:
        ipot = CompositePotentialFunction(X, [WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J), ShieldedLattice(X, -2.5e-10, 2.5e-10, 1.74*0.5, 1e-10, n=5), ShieldedPotential(X, 0, 2.5e-10, 1.74, 1e-10)])
    elif typ == 7:
        ipot = Potential(X)
    elif typ == 8:
        ipot = ShieldedPotential(X, 0, 2.5e-10, 1.74, 1e-10)

    if detPos==0:
        detPos = 0.9*X[-1]


    if typ != 7 and typ != 8:
        lf = LightField(X, Emax, lam, phase, env, WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J).zim, detPos, cent)
    elif typ == 8:
        lf = LightField(X, Emax, lam, phase, env, X[0]+bufferSize, detPos, cent)
    else:
        lf = LightField(X, 0, lam, phase, env, WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J).zim, detPos, cent)
    
    pot = PotentialFunction(X, ipot, lf)
    
    psiSampPoint = int(np.searchsorted(X, detPos))
    
#    env = Envelope()
##    lf = ConstantField(X, Emax, env, X[-1], X[0])
#    lf = LightField(X, Emax, np.inf, np.pi, env, X[0], X[-1], cent)
#    ipot = SquareWell(X, 1*eV_J, L)
#    pot = PotentialFunction(X, ipot, lf, keepInitialPotential=False)
    
    waveFunc = WaveFunction(pot, bufferSize, bufferLeft, tprec=tprec)
    if not initialState is 0:
        waveFunc.psi = initialState
        waveFunc.normalize()
    #tMax = oscToSave/c*lam
    #tMax = 2*cent+getIdealMaxT(X[-1], 1*eV_J)
    tMax = totTime
    ts = []
    vs = []
    acs = []
    ac2s = []
    xs = []
    xfs = []
    vfs = []
    psis = []
    psits = []
    lps = []
    EsT = []
    totProb = []
    sampfunc = []
    sampcur = []
    finalfunc = 0
    V0 = waveFunc.iV
    
    def saveData():
        Vwell = 1
        Es = EsT
        if savePsi:
            np.savez_compressed(fil, ts=ts, Es=Es, vs=vs, acs=acs, ac2s=ac2s, xs=xs, psis=psis, lps=lps, X=X, psits=psits, EsT=EsT, lam=lam, totProb=totProb, Emax=Emax, Vwell=Vwell, L=0, V0=V0, xfs=xfs, vfs=vfs, sampfunc=sampfunc, tau=tau, sampcur=sampcur, finalfunc=finalfunc, vdpos=detPos)
        else:
            np.savez_compressed(fil, ts=ts, Es=Es, vs=vs, acs=acs, ac2s=ac2s, xs=xs, psis=0, lps=lps, X=X, psits=0, EsT=EsT, lam=lam, totProb=totProb, Emax=Emax, Vwell=Vwell, L=0, V0=V0, xfs=xfs, vfs=vfs, sampfunc=sampfunc, tau=tau, sampcur=sampcur, finalfunc=finalfunc, vdpos=detPos)
    
    
    global cont
    cont = False
    if plotSaves and (platform == 'win32' or platform == 'cygwin'):
        global pause
        pause = False
        fig, ((ax1,ax2),(ax5,ax6), (ax3, ax4)) = plt.subplots(3,2)
        plt.tight_layout()
        fig.suptitle("$\lambda$ = %d nm, dt = %g s, dx = %g m"%(lam*1e9, waveFunc.dt, waveFunc.dx))
        lineProb, = ax1.plot(waveFunc.X, waveFunc.prob(), 'b-')
        ax1b = ax1.twinx()
        lineV, = ax1b.plot(pot.X, V0/eV_J, 'k--')
        lineJ, = ax1.plot(waveFunc.X, waveFunc.probCurrent(), 'g-')
        lineEL, = ax1b.plot([min(waveFunc.X), max(waveFunc.X)], [0,0], 'k:')
        ax1.grid()
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('P (m$^{-1}$), J (arb.u.)')
        ax1b.set_ylabel('V (eV)')
        #ax1b.set_ylim(-Vwell/eV_J/10.0, Vwell/eV_J*11/10.0)
        ax1.set_title("Wave Function")
        
        #lineE, = ax2.plot(ts, Es, 'b-')
        lineET, = ax2.plot(ts, EsT, 'k-')
        ax2b = ax2.twinx()
        lineLaserPhase1, = ax2b.plot(ts, EsT, 'g--')
        ax2.grid()
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('E (eV)')
        ax2.set_title("Energy")

#        lineJS, = ax2.plot(ts, EsT, 'k-')
#        ax2b = ax2.twinx()
#        lineLaserPhase1, = ax2b.plot(ts, EsT, 'g--')
#        ax2.grid()
#        ax2.set_xlabel('t (s)')
#        ax2.set_ylabel('J (s$^{-1}$)')
#        ax2.set_title("Probability Current Sample")
        
        lineFFT, = ax3.plot(0, 0, 'k-')
        ax3.set_xlabel('E (eV)')
        ax3.set_ylabel('log(I)')
        ax3.set_title('FFT Power Spectrum')
        
        linePSN, = ax6.plot(0,0,'k-')
        ax6.set_xlabel('<x> (m)')
        ax6.set_ylabel('<v> (m/s)')
        ax6.set_title("Phase Space")
        
        linePSF, = ax5.plot(0,0,'k-')
        ax5.set_xlabel('<x> (m)')
        ax5.set_ylabel('<v> (m/s)')
        ax5.set_title("Phase Space in Light Field")
        ax5.set_xlim(X[0], X[-1])
        ax5.grid()
        
        lineA, = ax4.plot(ts, acs, 'k-')
        ax4b = ax4.twinx()
        lineLaserPhase2, = ax4b.plot(ts, EsT, 'g--')
        ax4.grid()
        ax4.set_xlabel('t (s)')
        ax4.set_ylabel('<a$^2$> (m$^2$s$^{-4}$)')
        ax4.set_title("Expectation Acceleration Squared")
        
        def init():
            ax1.set_ylim(-4e9, 4e9)
            ax1.set_xlim(min(X), max(X))
            ax2b.set_ylim(-1, 1)
            ax4b.set_ylim(-1, 1)
            return lineProb, lineV, lineJ, lineET, lineA, lineFFT, lineLaserPhase1, lineLaserPhase2, linePSN, linePSF, lineEL

        def onClick(event):
            global pause
            pause = not pause
        fig.canvas.mpl_connect('key_press_event', onClick)
        
        def run(data):
            if not pause:
                X, psi2, V, J, t, a, a2, x, v, lp, et, ce, xfs, vfs = data
                lineProb.set_data(X, psi2)
                lineV.set_data(X, V/eV_J)
                lineJ.set_data(X, J/max(abs(J))*1e9)
                lineEL.set_data([min(X), max(X)], [ce,ce])
#                lineET.set_data(t, et)
                lineA.set_data(t, a2)
                linePSN.set_data(x, v)
                linePSF.set_data(xfs, vfs)
                if len(a)>2:
                    fftx, ffty = getFFTPowerSpectrum(a, waveFunc.dt)
                    lineFFT.set_data(fftx*hbar*2*np.pi/eV_J, np.log(ffty))
                    ax3.set_xlim(0, max(fftx)*hbar*2*np.pi/eV_J)
                    ax3.set_ylim(min(np.log(ffty)), max(np.log(ffty)))
                    
                lineLaserPhase1.set_data(t, lp)
                lineLaserPhase2.set_data(t, lp)
                if len(t)>2:
                    ax2.set_xlim(0, t[-1])
#                    ax2.set_ylim(min(et), max(et))
                    ax2.set_ylim(min(a), max(a))
                    ax4.set_xlim(0, t[-1])
                    ax4.set_ylim(min(a2), max(a2))
                    ax2b.set_ylim(-1, 1)
                    ax4b.set_ylim(-1, 1)
                    ax6.set_xlim(min(x), max(x))
                    ax6.set_ylim(min(v), max(v))
                    ax5.set_ylim(min(vfs), max(vfs))
            return lineProb, lineV, lineJ, lineET, lineA, lineFFT, lineLaserPhase1, lineLaserPhase2, linePSN, linePSF, lineEL
        
        def data_gen():
            global pause
            global cont
            while waveFunc.rcurT < tMax or cont or not stopWhenFinished:
                if not waveFunc.grounded and getGroundState:
                    err = waveFunc.findGroundState(itsBetweenSave)
                    waveFunc.grounded = err/eV_J<1e-24*itsBetweenSave
                    if waveFunc.grounded:
                        if printProgress:
                            print("Grounded")
#                        for i in range(500):
#                            waveFunc.stepCN(changeTime=False)
#                            waveFunc.decayEdge()
                    elif printProgress:
                        print("%.5f%% Grounded"%(1e-21*eV_J*100/err*itsBetweenSave))
                    curE = waveFunc.getEnergy()/eV_J
                    yield waveFunc.X, waveFunc.prob(), waveFunc.curV, waveFunc.probCurrent(), ts, acs, ac2s, xs, vs, lps, EsT, curE, xfs, vfs
                else:
                    if not pause:
                        for i in range(itsBetweenSave):
                            waveFunc.stepCN()
                            waveFunc.decayEdge()
                            ts.append(waveFunc.rcurT)
                            curE = waveFunc.getEnergy()/eV_J
                            EsT.append(curE)
                            vs.append(waveFunc.expectP()/me)
                            xs.append(waveFunc.expectX())
                            acs.append(waveFunc.expectA())
                            ac2s.append(waveFunc.expectA2())
                            lps.append(np.cos(lf.getPhase(waveFunc.rcurT))*env.getMax(waveFunc.rcurT))
                            totProb.append(waveFunc.totalProb())
                            xfs.append(waveFunc.expectXField())
                            vfs.append(waveFunc.expectPField()/me)
                            sampfunc.append(waveFunc.psi[psiSampPoint])
                            sampcur.append(waveFunc.probCurrentPt(psiSampPoint))
                            if savePsi and waveFunc.rcurT/tMax*savePsiRes>len(psis):
                                psis.append(reduceSize(abs(waveFunc.psi), savePsiRes))
                                psits.append(waveFunc.rcurT)
                        if saveInMiddle:
                            saveData()
                            
                        print(totProb[-1])
                        
                    yield waveFunc.X, waveFunc.prob(), waveFunc.curV, waveFunc.probCurrent(), ts, acs, ac2s, xs, vs, lps, EsT, curE, xfs, vfs
            else:
                if saveWhenFinished:
                    finalfunc = waveFunc.psi
                    saveData()
                if printProgress:
                    input("Simulation finished. Press Enter to continue calculations.")
                cont = True
                    
        ani = animation.FuncAnimation(fig, run, data_gen, init_func = init, interval=0)
        plt.show()
        plt.tight_layout()
    else:
        while waveFunc.rcurT < tMax or not stopWhenFinished:
            if not waveFunc.grounded and getGroundState:
                err = waveFunc.findGroundState(itsBetweenSave)
                waveFunc.grounded = err/eV_J<1e-24*itsBetweenSave
                #print("%.5f%% Grounded"%(1e-10*100/err*itsBetweenSave))
#                if waveFunc.grounded:
#                    for i in range(500):
#                        waveFunc.stepCN(changeTime=False)
#                        waveFunc.decayEdge()
            else:
                for i in range(itsBetweenSave):
                    waveFunc.stepCN()
                    waveFunc.decayEdge()
                    ts.append(waveFunc.rcurT)
                    EsT.append(waveFunc.getEnergy()/eV_J)
                    vs.append(waveFunc.expectP()/me)
                    xs.append(waveFunc.expectX())
                    acs.append(waveFunc.expectA())
                    ac2s.append(waveFunc.expectA2())
                    lps.append(np.cos(lf.getPhase(waveFunc.rcurT))*env.getMax(waveFunc.rcurT))
                    totProb.append(waveFunc.totalProb())
                    xfs.append(waveFunc.expectXField())
                    vfs.append(waveFunc.expectPField()/me)
                    sampfunc.append(waveFunc.psi[psiSampPoint])
                    sampcur.append(waveFunc.probCurrentPt(psiSampPoint))
                    if savePsi and waveFunc.rcurT/tMax*savePsiRes>len(psis):
                        psis.append(reduceSize(abs(waveFunc.psi), savePsiRes))
                        psits.append(waveFunc.rcurT)
                if saveInMiddle:
                    saveData()
                if printProgress:
                    print("%d nm:%3.2f%%"%(int(lam*1e9),min([(waveFunc.rcurT/tMax*100),100])))
        else:
            if saveWhenFinished:
                finalfunc = waveFunc.psi
                saveData()
            if printProgress:
                print("%d nm simulation finished."%(int(lam*1e9)))

if platform == 'win32' or platform == 'cygwin':
    def loadData(fil=None):
        """Loads data from a file.
        
        Args:
            fil (file, optional): File to open. If omitted, a prompt will ask for a file. Defaults to None.
            
        Returns:
            ts, Es, vs, acs, ac2s, xs, psis, lps, X, psits, EsT, lam, totProb, Emax, Vwell, L, V0, xfs, vfs, sampfunc, tau
        """
        if fil == None:
            root = tk.Tk()
            root.withdraw()
            fil = filedialog.askopenfile()
        if type(fil) == str:
            data = np.load(fil)
            print(os.path.basename(fil))
        else:   
            data = np.load(fil.name)
            print(os.path.basename(fil.name))
    
        ts = data['ts']
        Es = data['Es']
        vs = data['vs']
        acs = data['acs']
        ac2s = data['ac2s']
        xs = data['xs']
        psis = data['psis']
        lps = data['lps']
        X = data['X']
        psits = data['psits']
        EsT = data['EsT']
        lam = data['lam']
        totProb = data['totProb']
        Emax = data['Emax']
        Vwell = data['Vwell']
        L = data['L']
        V0 = data['V0']
        sampfunc = data['sampfunc']
        tau = data['tau']
        sampcur = 0#data['sampcur']
        finalfunc = 0#data['finalfunc']
        vdpos = 0#data['vdpos']
        
        xfs = data['xfs']
        vfs = data['vfs']
        
        data.close()
        
        return ts, Es, vs, acs, ac2s, xs, psis, lps, X, psits, EsT, lam, totProb, Emax, Vwell, L, V0, xfs, vfs, sampfunc, tau, sampcur, finalfunc, vdpos

    
    def plotFile(fil=None, show=True, getObjects=False, colorCodePhaseSpace=False, hqDensity=False, discreteColors=False, contours=True, logScaleTDP=True, densityContrast=1/np.e, photonContrast=1/2, electronContrast=1/2):
        """Plots a data file in a results summary format.
        
        Args:
            fil (file, optional): File to plot. If no file is given, a prompt will be provided. Defaults to None.
            show (bool, optional): Whether or not to show the plot. Defaults to True.
            getObjects (bool, optional): Whether or not to return the plot objects (figs and axes). Defaults to False.
            colorCodePhaseSpace (bool, optional): Whether or not to apply a color gradient to the phase space. Defaults to False.
            hqDensity (bool, optional): Whether or not to use a gourad method to the probability density plot. Defaults to False.
            discreteColors (bool, optional): Whether or not to use discretized colors in plotting the probability density. Defaults to False.
            contours (bool, optional): Whether or not to outline the probability density plot with contours. Defaults to True.
            logScaleTDP (bool, optional): Whether or not to use a log-scaled energy axis for time dependent electron and photon spectra. Defaults to True.
            densityContrast (float, optional): Contrast in the probability density plot. Defaults to 1/e.
            photonContrast (float, optional): Contrast in the time dependent photon spectrum. Defaults to 1/2.
            electronContrast (float, optional): Contrast in the time dependent electron spectrum. Defaults to 1/2.
            
        Returns:
            fig, axes if getObjects
                ax1: Initial potential function attached to probability density plot.
                ax2: Time dependent electron energy spectrum.
                ax3: Photon energy spectrum.
                ax4: Time dependent photon energy spectrum.
                ax5: Probability density plot.
                ax6: Light phase and expectation kinematics below probability density plot.
                ax7: Electron energy spectrum.
        """
        if fil == None:
            ts, Es, vs, acs, ac2s, xs, psis, lps, X, psits, EsT, lam, totProb, Emax, Vwell, L, V0, xfs, vfs, sampfunc, tau, sampcur, finalfunc, vdpos = loadData()
        else:
            ts, Es, vs, acs, ac2s, xs, psis, lps, X, psits, EsT, lam, totProb, Emax, Vwell, L, V0, xfs, vfs, sampfunc, tau, sampcur, finalfunc, vdpos = loadData(fil)
        #fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4,2)
        fig = plt.figure()
        gs0 = gridspec.GridSpec(2, 1, left=0.05, right=0.95, bottom=0.05)
        gs00 = gridspec.GridSpecFromSubplotSpec(4, 12, subplot_spec=gs0[0], wspace=0, hspace=0)
        gs01 = gridspec.GridSpecFromSubplotSpec(4, 32, subplot_spec=gs0[1], wspace=0, hspace=0)
        
        ax5 = plt.Subplot(fig, gs00[:3,1:11])
        fig.add_subplot(ax5)
        #ax5c = plt.Subplot(fig, gs0[0:4,-1])
        #fig.add_subplot(ax5c)
        ax1 = plt.Subplot(fig, gs00[:3, 0], sharey=ax5)
        fig.add_subplot(ax1)
        ax6 = plt.Subplot(fig, gs00[3, 1:11], sharex=ax5)
        fig.add_subplot(ax6)
        ax5c = plt.Subplot(fig, gs00[:, 11])
        
        ax4 = plt.Subplot(fig, gs01[0:2, :15])
        fig.add_subplot(ax4)
        ax2 = plt.Subplot(fig, gs01[2:4, :15], sharex=ax4)
        fig.add_subplot(ax2)
        ax3 = plt.Subplot(fig, gs01[0:2, 18:])
        fig.add_subplot(ax3)
        ax7 = plt.Subplot(fig, gs01[2:4, 18:])
        fig.add_subplot(ax7)
        ax8 = plt.Subplot(fig, gs01[1:3, :15], sharex=ax4)
        fig.add_subplot(ax8)
        ax4c = plt.Subplot(fig, gs01[0:2, 15])
        #fig.add_subplot(ax4c)
        ax2c = plt.Subplot(fig, gs01[2:4, 15])
        #fig.add_subplot(ax2c)
        
        maxElec = getPond(Emax, lam)*10/eV_J*1.3+50
        maxPhot = getPond(Emax, lam)*3.17/eV_J*1.2+50
        dt = (ts[-1]-ts[0])/len(ts)
        
        sampfunc = sampfunc.conj()
        
        fig.suptitle(r"$\lambda$ = %d nm, $\tau$ = %g fs, $E_{max}$ = %g GVm$^{-1}$, $\gamma_{Keldysh}$ = %f, $U_{p}$ = %g eV, dt = %g fs, dx = %g $\AA$"%(lam*1e9, tau*1e15, Emax*1e-9, getKeldysh(6.2*eV_J, Emax, lam), getPond(Emax, lam)/eV_J, (ts[1]-ts[0])*1e15, (X[1]-X[0])*1e10))
        if psis.size > 1:
            v = np.log10(psis.transpose())
            lastFunc = v[:,-1]
            mn = lastFunc.mean()
            stdev = lastFunc.std()
            if contours:
                msh = ax5.contourf(psits*1e15, np.linspace(X[0], X[-1], v.shape[0])*1e10, v, levels=np.linspace(mn-stdev/photonContrast, mn+stdev/photonContrast, 30, endpoint=True), cmap=("tab20" if discreteColors else "magma"), shading='gouraud' if hqDensity else 'flat', extend="both")
            else:
                msh = ax5.pcolormesh(psits*1e15, np.linspace(X[0], X[-1], v.shape[0])*1e10, v, cmap=("tab20" if discreteColors else "magma"), vmin=mn-stdev/densityContrast, vmax=mn+stdev/densityContrast, shading='gouraud' if hqDensity else 'flat')

            #ax5.set_xlabel('t (s)')
            #ax5.set_ylabel('x (m)')
            ax5.yaxis.set_visible(False)
            ax5.xaxis.set_visible(False)
            ax5.set_title('Probability Density')
            ax5c = fig.colorbar(msh, ax=ax5c, format="%.1f")
            ax5c.set_label(r'$\log_{10}(\rho$)')
        else:
            if colorCodePhaseSpace:
                pts = np.array([xs, vs]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=plt.get_cmap('brg'), norm=plt.Normalize(ts[0], ts[-1]))
                lc.set_array(ts)
                ax5.add_collection(lc)
                ax5.set_xlim(min(xs), max(xs))
                ax5.set_ylim(min(vs), max(vs))
                #ax5c = fig.colorbar(lc, ax=ax5)
#                ax5c.set_label('t (fs)')
                linePS = None
            else:
                linePS, = ax5.plot(xs, vs, 'k-')
            ax5.set_xlabel('<$x_{net}$> (m)')
            ax5.set_ylabel('<$v_{net}$> (m/s)')
            ax5.set_title('Expectation Value Phase Space')

        fftx, ffty = getFFTPowerSpectrum(acs, dt)
        fftx = np.array(fftx)*4.13567e-15
        ip = np.searchsorted(fftx, maxPhot)
        lineFFT, = ax3.plot(fftx, np.log10(ffty), 'k-')
        ax3.set_xlabel('E (eV)')
        ax3.set_ylabel(r'$\log_{10}$(I) (arb. u.)')
        ax3.set_title('Photon Spectrum', loc='right', y=0.85)
        ax3.grid(axis='x')
        ax3.xaxis.tick_top()
        ax3.xaxis.set_label_position('top')
        ax3.set_xlim(0, maxPhot)
        ax3.set_ylim(min(np.log10(ffty[:ip])), max(np.log10(ffty[:ip])))
        
        
#        fftx, ffty = getFFTPowerSpectrum(psisamps, ts[1]-ts[0])
#        fftx = np.array(fftx)*4.13567e-15
#        lineFFT2, = ax7.plot(fftx, np.log(ffty), 'k-')
#        ax7.set_xlim(0, max(fftx))
#        ax7.set_xlabel('E (eV)')
#        ax7.set_ylabel('log(I) (arb. u.)')
#        ax7.set_title('Electron Spectrum', loc='right', y=0.85)
#        ax7.grid(axis='x')

#        Edet = np.clip(hbar**2*dphase**2/(2*me*eV_J), 0, max(fftx))
#        hist,_ = np.histogram(Edet, bins=int(len(fftx)), weights=sampprob)
#        x = np.linspace(0, max(Edet), len(hist))
#        lineEE, = ax7.plot(x, np.log10(hist), 'k-')
        fftx, ffty = getFFTPowerSpectrum(sampfunc, dt)
        fftx = np.array(fftx)*4.13567e-15
        ie = np.searchsorted(fftx, maxElec)
        lineEE, = ax7.plot(fftx, np.log10(ffty), 'k-')
        ax7.set_xlabel('E (eV)')
        ax7.set_ylabel(r'$\log_{10}$(I) (arb. u.)')
        ax7.set_title('Electron Spectrum', loc='right', y=0.85)
        ax7.grid(axis='x')
        ax7.set_xlim(0, maxElec)
        ax7.set_ylim(min(np.log10(ffty[:ie])), max(np.log10(ffty[:ie])))
        
#        ax4.plot(ts*1e15, ac2s, 'k-')
#        ax4b = ax4.twinx()
#        ax4b.plot(ts*1e15, lps, 'g--')
#        ax4.set_xlabel('t (fs)')
#        ax4.set_ylabel('<a$^2$> (m$^2$s$^{-4}$)')
#        ax4b.set_ylabel('$\phi_{field}$')
#        ax4.set_xlim(ts[0]*1e15, ts[-1]*1e15)
#        ax4.set_title('Expectation Acceleration Squared', loc='right', y=0.85)
#        ax4b.yaxis.label.set_color('green')
#        ax4.grid()
#        ax4.tick_params(labelbottom='off')

#        fftys = []
#        N = len(ac2s)
#        tc = 1000
#        stp = int(N/tc)
#        sz = 10
#        for I in range(tc-sz+1):
#            fftx,ffty = getFFTPowerSpectrum(ac2s[I*stp:min((I+sz)*stp, N)], dt)
#            fftys.append(ffty)
#        fftx *= 4.13567e-15
#        fftx = fftx
#        fftys = np.log10(np.array(fftys).transpose())
#        cutoff = np.searchsorted(fftx, maxPhot)
#        mn = fftys[:cutoff, :].mean()
#        stdev = fftys[:cutoff, :].std()
        tc = 1000
        minPhot=1
        fftx = np.geomspace(4.13567e-15/(maxPhot*dt), 4.13567e-15/(minPhot*dt), 150)
        nts = np.linspace(ts[0], ts[-1], tc)
        def wavelet(npts, wid):
            x = np.linspace(-1, 1, npts)
            wid /= npts
            return np.exp(-(x/wid)**2/4)*np.sin(2*np.pi/wid*x)

        fftys = np.log10(waveletTransform(acs, wavelet, fftx, ts, tc) ** 2)
        fftys[np.isneginf(fftys)] = np.nanmin(fftys)
        fftys[np.isinf(fftys)] = np.nanmax(fftys)
        mn = fftys.mean()
        stdev = fftys.std()
        fftx = 4.13567e-15/dt/fftx
        mshacs = ax4.contourf(nts*1e15, fftx, fftys, cmap=PEcmap, levels=np.linspace(mn-stdev/photonContrast, mn+stdev/photonContrast, 30, endpoint=True), shading='gouraud' if hqDensity else 'flat', extend='both')
        if logScaleTDP:
            ax4.set_yscale("log")
        v = np.arange(int(mn-stdev/photonContrast), int(mn+stdev/photonContrast), int(2*stdev/photonContrast/10+1))
        ax4c = fig.colorbar(mshacs, ax=ax4c, format='%d', ticks=v)
        #ax4c.set_label(r'$\log_{10}(I)$ (arb. u.)')
        ax4.set_ylim(minPhot, maxPhot)
        ax4.set_ylabel('E (eV)')
        ax4.set_title('Photon Spectrum', loc='right', y=0.85)
        ax4.xaxis.set_visible(False)
        
        #ax2.plot(ts, Es, 'b-')
        #ax2.plot(ts*1e15, Edet*sampprob, 'k-')
        #( phases + np.pi) % (2 * np.pi ) - np.pi
#        ax2.plot(ts*1e15, np.gradient(np.unwrap(np.angle(sampfunc)), ts[1]-ts[0])*hbar/eV_J, 'k-')
#        ax2b = ax2.twinx()
#        ax2b.plot(ts*1e15, lps, 'g--')
#        ax2c = ax2.twinx()
#        ax2c.plot(ts*1e15, abs(sampfunc)**2, 'b-')
#        ax2.set_xlabel('t (fs)')
#        ax2.set_ylabel(r'$\frac{d}{dt}\phi_{d}$ (eV)')
#        ax2c.set_ylabel(r'$\rho_{d}$')
#        ax2c.yaxis.label.set_color('blue')
#        ax2b.yaxis.set_visible(False)
#        ax2.set_xlim(ts[0]*1e15, ts[-1]*1e15)
#        ax2.set_title('Sampled Wavefunction', loc='right', y=0.85)
#        ax2.grid()
        
#        N = len(ac2s)
#        stp = int(N/tc)
#
#        fftys = []
#        N = len(sampfunc)
#        stp = int(N/tc)
#        sz = 5
#        for I in range(tc-sz+1):
#            fftx,ffty = getFFTPowerSpectrum(sampfunc[I*stp:min((I+sz)*stp, N)], dt)
#            fftys.append(ffty)
#        fftx *= 4.13567e-15
#        fftx = fftx
#        fftys = np.log10(np.array(fftys).transpose())
#        cutoff = np.searchsorted(fftx, maxElec)
#        mn = fftys[:cutoff, :].mean()
#        stdev = fftys[:cutoff, :].std()
        minElec=1
        fftx = np.geomspace(4.13567e-15/(maxElec*dt), 4.13567e-15/(minElec*dt), 150)
        def wavelet(npts, wid):
            x = np.linspace(-1, 1, npts)
            wid /= npts
            return np.exp(-(x/wid)**2/4)*np.exp(1j*2*np.pi/wid*x)

        fftys = np.log10(abs(waveletTransform(sampfunc, wavelet, fftx, ts, tc))**2)
        fftys[np.isneginf(fftys)] = np.nanmin(fftys)
        fftys[np.isinf(fftys)] = np.nanmax(fftys)
        mn = fftys.mean()
        stdev = fftys.std()
        fftx = 4.13567e-15/dt/fftx
        mshacs = ax2.contourf(nts*1e15, fftx, fftys, cmap=PEcmap, levels=np.linspace(mn-stdev/electronContrast, mn+stdev/electronContrast, 30, endpoint=True), shading='gouraud' if hqDensity else 'flat', extend='both')
        if logScaleTDP:
            ax2.set_yscale("log")
        v = np.arange(int(mn-stdev/electronContrast), int(mn+stdev/electronContrast), int(2*stdev/electronContrast/10+1))
        ax2c = fig.colorbar(mshacs, ax=ax2c, format='%d', ticks=v)
        #ax2c.set_label(r'$\log_{10}(I)$ (arb. u.)')
        ax2.set_ylim(minElec, maxElec)
        
        ax2.set_xlabel('t (fs)')
        ax2.set_ylabel('E (eV)')
        ax2.set_title('Electron Spectrum', loc='right', y=0.85)
#        ax6.plot(ts, totProb, 'k-')
#        ax6b = ax6.twinx()
#        ax6b.plot(ts, lps, 'g--')
#        ax6.set_xlim(min(ts), max(ts))
#        ax6b.set_ylabel('$\phi_{field}$')
#        ax6.set_xlabel('t (s)')
#        ax6.set_ylabel('$P_{tot}$')
#        ax6.set_title('Total Probability in System')
#        ax6b.yaxis.label.set_color('green')
#        ax6.grid()
    
        ax1.plot(V0/eV_J, X*1e10, 'k-')
        ax1.set_ylabel('x, <$x_{lf}$> ($\AA$)')
        ax1.set_xlabel('V (eV)', horizontalalignment='right', fontsize=10) 
        ax1.tick_params(axis='x', labelsize=6)
        #ax1.set_title('Initial Potential Function and Phase Space in Light Field')
        ax1b = ax1.twiny()
        if colorCodePhaseSpace:
            pts = np.array([vfs, xfs*1e10]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=plt.get_cmap('brg'), norm=plt.Normalize(ts[0], ts[-1]))
            lc.set_array(ts)
            ax1b.add_collection(lc)
            linePSF = None
        else:
            linePSF, = ax1b.plot(vfs, xfs*1e10, 'b-')
            ax1b.yaxis.label.set_color('blue')
        #ax1b.set_xlabel('<$v_{lf}$> (m/s)')
        ax1b.xaxis.set_visible(False)
        ax1b.set_xlim(min(vfs), max(vfs))
        ax1.set_ylim(X[0]*1e10, X[-1]*1e10)
        ax1.set_xlim(min(V0/eV_J), max(V0/eV_J))
        ax1.invert_xaxis()
        
        def fitFunc(x, mul, off):
            return lps*mul+off
        res, _ = scipy.optimize.curve_fit(fitFunc, lps, acs, p0=[(max(acs) - min(acs)) / 2, acs.mean()])
        newacs = highpass_filter(acs - fitFunc(lps, res[0], res[1]), 1 / (ts[1] - ts[0]), c / lam * 8, c / lam * 4)
        ax6d = ax6.twinx()
        l4 = ax6d.plot(np.linspace(ts[0], ts[-1], len(newacs))*1e15, newacs, 'b-', label = '<a$^2$>$_{hpf}$', linewidth=1)
        ax6d.set_ylim(min(newacs*np.hanning(len(newacs)))*1.5,max(newacs*np.hanning(len(newacs)))*1.5)
        ax6d.yaxis.set_visible(False)
        l1 = ax6.plot(ts*1e15, xs*1e10, 'k-', label = '<x>')
        ax6b = ax6.twinx()
        l2 = ax6b.plot(ts*1e15, vs, 'r-', label = '<v>')
        ax6c = ax6.twinx()
        ax6c.yaxis.set_visible(False)
        l3 = ax6c.plot(ts*1e15, lps, 'g--', label = '$\phi$')
        ax6b.set_ylabel('<v> (m/s)')
        ax6.set_xlabel('t (fs)')
        ax6.set_ylabel('<x> ($\AA$)', verticalalignment='bottom', fontsize=8)
        ax6.tick_params(axis='y', labelsize=6)
        #ax6.set_title('Position and Velocity')
        ax6b.yaxis.label.set_color('red')
        ax6.grid()
        lns = l1+l2+l3+l4
        labs = [l.get_label() for l in lns]
        ax6.legend(lns, labs, loc=9, ncol=4, borderaxespad=0)
        ax6.set_xlim(ts[0]*1e15, ts[-1]*1e15)
        
        ax8.patch.set_alpha(0)
        ax8.plot(ts*1e15, lps, 'r--', alpha=0.6)
        ax8.set_xlim(ts[0]*1e15, ts[-1]*1e15)
        ax8.axis('off')
        #ax6.plot(ts, totProb, 'k-')

        fig.set_size_inches(18,9)
        #plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if show:
            plt.show()
        
        if getObjects:
            return (fig, ax1, ax2, ax3, ax4, ax5, ax6, ax7)

def testDX(dx):
    """Used to test the effects of different dx values.
    
    Args:
        dx (float): dx value to test.
    
    Returns:
        float: Ratio of final acceleration to initial acceleration (1-err)
        float: Total probability at end of test.
        float: Final expectation value of x.
        float: Final expectation value of p.
    """
    L = 4e-10
    X = np.arange(-30, 30.0, dx)*L
    Vwell = 6*eV_J
    lam = 1000e-9
    Emax = 1e10
    bufferSize = 5*L
    
    env = Envelope()
    lf = LightField(X, Emax, lam, np.pi/2, env, X[0], X[-1], 0)
    ipot = SquareWell(X, Vwell, L)
    pot = PotentialFunction(X, ipot, lf, keepInitialPotential=False)
    waveFunc = WaveFunction(pot, bufferSize, True)
    while not waveFunc.grounded:
        waveFunc.grounded = waveFunc.findGroundState(100)<1e-9
    for i in range(10):
        waveFunc.stepCN(1)
    a0 = waveFunc.expectA()
    p = waveFunc.expectP()
    x = waveFunc.expectX()
    while x < 15*L and p > -1e-26:
        for i in range(int(1/dx/10)):
            waveFunc.stepCN(1)
        x = waveFunc.expectX()
        p = waveFunc.expectP()
    return waveFunc.expectA()/a0, waveFunc.totalProb(), x, p
        
def testEff():
    """Used to test the efficiency of various functions."""
    cent = 20e-15
    env = Pulse(5e-15, cent)
    X = np.arange(-20e-9, 40e-9, 1e-13)
    lf = LightField(X, 1e10, 800e-9, 3*np.pi/2, env, 0, max(X), cent)
    ipot = WachterComposite(X, 9.2*eV_J, 6.2*eV_J)
    pot = PotentialFunction(X, ipot, lf)
    wav = WaveFunction(pot, 5e-9, True)
    av = 0
    for i in range(100):
        t = time.time()
        wav.expectP()
        av += time.time()-t
    return av/100

#plot lots of files with modified plot settings
def plotFilesMod(discColor=False, contours=True, fils=None, desc=None):
    """Plots and saves a lot of files. Axes may be edited within this function without messing with plotFile.
    Will pop up with a window asking for files to plot.
    A descriptor may be added, appended to the original data file name.
    Images will be saved as their corresponding file name plus the descriptor in the same folder as the files.
    The process is not parallelized, so consider running multiple instances of this function.
    
    Args:
        discColor (bool, optional): Whether or not to use discrete colors. Defaults to False.
        contours (bool, optional): Whether or not to use contours in the probability distribution as opposed to a smooth plot. Defaults to True.
        fils (array, optional): List of files to plot. If left none, a popup will ask for a selection of files. Defaults to None.
        desc (str, optional): Description to apply to files. If left none, will prompt for a description. Defaults to None.
    """
    if fils is None:
        root = tk.Tk()
        root.withdraw()
        fils = filedialog.askopenfiles()
    if desc is None:
        desc = input("Output files' descriptor: ")
    for fil in fils:
        (fig, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plotFile(fil, show=False, getObjects=True, colorCodePhaseSpace=True, hqDensity=True, discreteColors=discColor, contours=contours)
#        ax3.set_xlim(0, 170)
#        ax7.set_xlim(0, 700)
#        ax4.set_ylim(0, 170)
#        ax2.set_ylim(0, 700)
        #ax7.set_ylim(-4, 12)
        plt.savefig(os.path.splitext(fil.name)[0]+desc+".png", dpi=500)
        plt.close()
        del fig, ax1, ax2, ax3, ax4, ax5, ax6, ax7
    
def plotVarEnergySpectra(LNE=True, PNE=True, emax=100, w=6.2, Ef=9.2, printProgress=True):
    """Plots the energy spectra with a variable (wavelength or electric field strength) changing.
    Will ask to provide a set of files whose wavelengths or electric field strengths are changing.
    Will ask to provide a file name to save image as.
    
    Args:
        LNE (bool, optional): Lambda not E-field, True means lambda is chaning, False means the max e-field is changing. This depends on the files in question. Defaults to True.
        PNE (bool, optional): Photon not Electron, True will plot the photon spectra, False, will plot the electron spectra. Defaults to True.
        emax (float, optional): Maximum energy to plot, eV. This should correspond to about 3.17Up for photons and 10Up for electrons. Defaults to 100 eV.
        w (float, optional): Work function of metal, eV. Defaults to 6.2 eV.
        Ef (float, optional): Fermi energy of metal, eV. Defaults to 9.2 eV.
        printProgress (bool, optional): Whether or not to print the progress. Defaults to True.
    """
    root = tk.Tk()
    root.withdraw()
    fils = filedialog.askopenfiles()
    sfil = filedialog.asksaveasfile(mode='w', defaultextension='.png')
    xdat = []
    zdat = []
    Ups = []
    energ = np.linspace(0, emax, 1000)
    if PNE:
        fftdat = 'acs'
        titl = 'Photon Spectrum, '
    else:
        fftdat = 'sampfunc'
        titl = 'Electron Spectrum, '
    
    data = np.load(fils[0].name)
    if LNE:
        x = 'lam'
        xmul = 1e9
        xlab = '$\lambda$ (nm)'
        titl += '$E_{max}$ = %g GVm$^{-1}$'%(data['Emax']*1e-9)
        
    else:
        x = 'Emax'
        xmul = 1e-9
        xlab = '$E_{max}$ (GVm$^{-1}$)'
        titl += '$\lambda$ = %g nm'%(data['lam']*1e9)
    data.close()
        
    i = 0

    for fil in fils:
        i += 1
        data = np.load(fil.name)
        xdat.append(data[x]*xmul)
        fftval = data[fftdat]
        if not PNE:
            fftval = fftval.conj()
        ts = data['ts']
        fftx, ffty = getFFTPowerSpectrum(fftval, (ts[-1]-ts[0])/len(ts))
        fftx *= 4.13567e-15
        zdat.append(np.log10(np.interp(energ, fftx, ffty)))
        
        lps = data['lps']
        Emaxeff = data['Emax']*max(lps)
        tlo = ts[lps.argmin()]
        thi = ts[lps.argmax()]
        lameff = c*2*(thi-tlo)
        Ups.append(getPond(Emaxeff, lameff))
#        Ups.append(getPond(data['Emax']*max(data['lps']), data['lam']))
        data.close()
        print("%3.2f%%"%(i/len(fils)*100), end="\r")
    
    #del fils
    
    Ups = np.array(Ups)
    if PNE:
        Ups *= 3.17
    else:
        Ups *= 10
    
    fig, ax1 = plt.subplots(1,1)
    cl = ax1.contourf(xdat, energ, np.array(zdat).T, 30, cmap=PEcmap, shading='gourad')
#    cl = ax1.pcolormesh(xdat, energ, np.array(zdat).T, cmap=PEcmap, shading='gourad')
    ax1.plot(xdat, Ups/eV_J+(20 if PNE else 0), 'r--')
    cb = plt.colorbar(cl)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel('$E$ (eV)')
    cb.set_label('$\log_{10}(I)$ (arb. u.)')
    ax1.set_ylim(0, emax)
    ax1.set_title(titl)
    #ax1.set_xlim(0, emax)
    #plt.show()
    plt.savefig(sfil.name, dpi=250)
    del sfil

def plotEnergySpectra(phote=100, elece=300):
    """Plots the photon and electron energy spectra from a file.
    Will ask to provide a data file.
    Will ask to provide a file name to save image as.
    
    Args:
        phote (float, optional): Maximum energy to plot in the photon spectrum, eV. This should correspond to about 3.17Up+I. Defaults to 100 eV.
        elece (float, optional): Maximum energy to plot in the electron spectrum, eV. This should correspond to about 10Up. Defaults to 300 eV.
    """
    
    root = tk.Tk()
    root.withdraw()
    fil = filedialog.askopenfile()
    sfil = filedialog.asksaveasfile(mode='w', defaultextension='.png')
    
    data = np.load(fil.name)
    acs = data['acs']
    funcs = np.conj(data['sampfunc'])
    ts = data['ts']
    photx, photy = getFFTPowerSpectrum(acs, (ts[-1]-ts[0])/len(ts))
    photy = np.log10(photy)
    photx *= 4.13567e-15
    elecx, elecy = getFFTPowerSpectrum(funcs, (ts[-1]-ts[0])/len(ts))
    elecy = np.log10(elecy)
    elecx *= 4.13567e-15
    fig, (ax1, ax2) = plt.subplots(2,1)
    
    ax1.plot(photx, photy, 'k-', lw=1)
    ax1.set_xlabel('E (eV)')
    ax1.set_ylabel(r'$\log_{10}$(I) (arb. u.)')
    ax1.set_title('Photon Spectrum')
    ax1.grid(axis='x')
    ie = np.searchsorted(photx, phote)
    ax1.set_xlim(0, phote)
    ax1.set_ylim(min(photy[:ie]), max(photy[:ie]))
    
    ax2.plot(elecx, elecy, 'k-', lw=1)
    ax2.set_xlabel('E (eV)')
    ax2.set_ylabel(r'$\log_{10}$(I) (arb. u.)')
    ax2.set_title('Electron Spectrum')
    ax2.grid(axis='x')
    ie = np.searchsorted(elecx, elece)
    ax2.set_xlim(0, elece)
    ax2.set_ylim(min(elecy[:ie]), max(elecy[:ie]))
    
    fig.suptitle(r'$\lambda=$%d nm, $E_{max}=$%d GVm$^{-1}$, $\tau=$%d fs'%(data['lam']*1e9, data['Emax']*1e-9, data['tau']*1e15))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    plt.savefig(sfil.name, dpi=500)

def plotElectronSpectrum(elece=300):
    """Plots the electron energy spectrum from a file.
    Will ask to provide a data file.
    Will ask to provide a file name to save image as.
    
    Args:.
        elece (float, optional): Maximum energy to plot in the electron spectrum, eV. This should correspond to about 10Up. Defaults to 300 eV.
    """
    root = tk.Tk()
    root.withdraw()
    fil = filedialog.askopenfile()
    sfil = filedialog.asksaveasfile(mode='w', defaultextension='.png')
    
    data = np.load(fil.name)
    funcs = np.conj(data['sampfunc'])
    ts = data['ts']
    elecx, elecy = getFFTPowerSpectrum(funcs, (ts[-1]-ts[0])/len(ts))
    elecy = np.log10(elecy)
    elecx *= 4.13567e-15
    fig, (ax1) = plt.subplots(1,1)
    ax1.plot(elecx, elecy, 'k-', lw=1)
    ax1.set_xlabel('E (eV)')
    ax1.set_ylabel(r'$\log_{10}$(I) (arb. u.)')
    ax1.set_title(r'Electron Spectrum: $\lambda=$%d nm, $E_{max}=$%d GVm$^{-1}$, $\tau=$%d fs'%(data['lam']*1e9, data['Emax']*1e-9, data['tau']*1e15))
    ax1.grid(axis='x')
    ie = np.searchsorted(elecx, elece)
    ax1.set_xlim(0, elece)
    ax1.set_ylim(min(elecy[:ie]), max(elecy[:ie]))
    plt.tight_layout()
    plt.savefig(sfil.name, dpi=500)

def plotPhotonSpectrum(phote=100):
    """Plots the photon energy spectrum from a file.
    Will ask to provide a data file.
    Will ask to provide a file name to save image as.
    
    Args:.
        phote (float, optional): Maximum energy to plot in the photon spectrum, eV. This should correspond to about 3.17Up. Defaults to 100 eV.
    """
    root = tk.Tk()
    root.withdraw()
    fil = filedialog.askopenfile()
    sfil = filedialog.asksaveasfile(mode='w', defaultextension='.png')
    
    data = np.load(fil.name)
    acs = data['acs']
    ts = data['ts']
    photx, photy = getFFTPowerSpectrum(acs, (ts[-1]-ts[0])/len(ts))
    photy = np.log10(photy)
    photx *= 4.13567e-15
    fig, (ax1) = plt.subplots(1,1)
    ax1.plot(photx, photy, 'k-', lw=1)
    ax1.set_xlabel('E (eV)')
    ax1.set_ylabel(r'$\log_{10}$(I) (arb. u.)')
    ax1.set_title(r'Photon Spectrum: $\lambda=$%d nm, $E_{max}=$%d GVm$^{-1}$, $\tau=$%d fs'%(data['lam']*1e9, data['Emax']*1e-9, data['tau']*1e15))
    ax1.grid(axis='x')
    ie = np.searchsorted(photx, phote)
    ax1.set_xlim(0, phote)
    ax1.set_ylim(min(photy[:ie]), max(photy[:ie]))
    plt.tight_layout()
    plt.savefig(sfil.name, dpi=500)

def compareElectronSpectrumMethods(elece=300):
    """Plots the electron energy spectrum from a file, using time-FFT, space-FFT, and binning.
    Will ask to provide a data file.
    Will ask to provide a file name to save image as.
    
    Args:
        elece (float, optional): Maximum energy to plot in the electron spectrum, eV. This should correspond to about 10Up. Defaults to 300 eV.
    """

    root = tk.Tk()
    root.withdraw()
    fil = filedialog.askopenfile()
    #sfil = filedialog.asksaveasfile(mode='w', defaultextension='.png')

    data = np.load(fil.name)

    fig, (ax2, ax1, ax3, ax4) = plt.subplots(4,1)

    #Time FFT
    funcs = np.conj(data['sampfunc'])
    ts = data['ts']
    elecx, elecy = getFFTPowerSpectrum(funcs, (ts[-1]-ts[0])/len(ts))
    elecy = np.log10(elecy/hbar)
    elecx *= 4.13567e-15
    ax1.plot(elecx, elecy, 'k-', lw=1)
    ax1.set_xlabel('E (eV)')
    ax1.set_ylabel(r'$\log_{10}$($\psi(E)$) (arb. u.)')
    ax1.set_title('Time FFT')
    ax1.grid(axis='x')
    ie = np.searchsorted(elecx, elece)
    ax1.set_xlim(0, elece)
    ax1.set_ylim(min(elecy[:ie]), max(elecy[:ie]))

    X = data['X']
    zim = WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J).zim
    zimind = int(np.searchsorted(X, zim))
    funcs = data['finalfunc']
    elecx, elecy = getFFT(funcs[zimind:], (X[-1]-X[0])/len(X)) #position to wave number space
    elecy = np.log10((abs(elecy)**2)*me/hbar**2)
    elecy = elecy[0:int(len(elecy)/2)]
    elecx = (elecx[0:int(len(elecx)/2)]*hbar*2*np.pi)**2/(2*me)/qe #convert wave number to energy, eV
    #elecx = elecx*hbar #convert wave number to momentum
    lin2, = ax2.plot(elecx, elecy, 'k-', lw=1)
    ax2.set_xlabel('E (eV)')
    ax2.set_ylabel(r'$\log_{10}$($\psi(E)$) (arb. u.)')
    ax2.set_title('Spatial FFT')
    ax2.grid(axis='x')
    ie = np.searchsorted(elecx, elece)
    ax2.set_xlim(0, elece)
    ax2.set_ylim(min(elecy[0:ie]), max(elecy[0:ie]))

    selmul = Slider(ax4, 'Space FFT Detection Zone', X[0], X[-2], valinit=zim)

    cur = data['sampcur']
    psi = data['sampfunc']
    elecx, elecy = getEnergySpecFromCurrent(cur, psi, 10000, 0, elece)
    elecy = np.log10(abs(elecy))
    lin3, = ax3.plot(elecx, elecy, 'k-', lw=1)
    ax3.set_title('Probability Current Histogram')
    ax3.set_xlabel('E (eV)')
    ax3.set_ylabel(r'$\log_{10}(P(E))$ (arb. u.)')
    ax3.grid(axis='x')
    ax3.set_xlim(0, elece)
    

    def update(val):
        zimind = int(np.searchsorted(X, val))
        elecx, elecy = getFFT(funcs[zimind:], (X[-1]-X[0])/len(X))
        elecy = np.log10((abs(elecy)**2)*me/hbar**2)
        elecy = elecy[0:int(len(elecy)/2)]
        elecx = (elecx[0:int(len(elecx)/2)]*hbar*2*np.pi)**2/(2*me)/qe
        lin2.set_data(elecx, elecy)
        ie = np.searchsorted(elecx, elece)
        ax2.set_ylim(min(elecy[0:ie]), max(elecy[0:ie]))
        fig.canvas.draw_idle()
    selmul.on_changed(update)

    fig.suptitle(r'Electron Spectrum: $\lambda=$%d nm, $E_{max}=$%d GVm$^{-1}$, $\tau=$%d fs'%(data['lam']*1e9, data['Emax']*1e-9, data['tau']*1e15))

    plt.tight_layout()
    #plt.savefig(sfil.name, dpi=500)
    plt.show()

def combineElectronSpectra(elece=300):
    """Plots the electron energy spectrum from a file, using time-FFT and space-FFT combined.
    Will ask to provide a data file.
    Will ask to provide a file name to save image as.
    
    Args:
        elece (float, optional): Maximum energy to plot in the electron spectrum, eV. This should correspond to about 10Up. Defaults to 300 eV.
    """

    root = tk.Tk()
    root.withdraw()
    fil = filedialog.askopenfile()
    #sfil = filedialog.asksaveasfile(mode='w', defaultextension='.png')

    data = np.load(fil.name)

    fig, (ax2, ax1, ax3, ax4) = plt.subplots(4,1)

    X = data['X']
    zim = WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J).zim
    zimind = int(np.searchsorted(X, zim))
    detind = int(np.searchsorted(X, data['vdpos']))

    tfunc = np.conj(data['sampfunc'])
    ts = data['ts']
    sfunc = data['finalfunc'][zimind:detind]
    telecx, telecy = getFFT(tfunc, (ts[-1]-ts[0])/len(ts), tukeyFlatWindow(len(tfunc), True))
    selecx, selecy = getFFT(sfunc, (X[-1]-X[0])/len(X), tukeyFlatWindow(len(sfunc), True, 1))
    
    telecx = telecx[0:int(len(telecx)/2)]*4.13567e-15
    telecy = telecy[0:int(len(telecy)/2)]/np.sqrt(hbar)

    selecx = (selecx[0:int(len(selecx)/2)]*hbar*2*np.pi)**2/(2*me*qe)
    selecy = selecy[0:int(len(selecy)/2)]*np.sqrt(me)/hbar

    #Time FFT
    ax1.plot(telecx, np.log10(abs(telecy)**2), 'k-', lw=1)
    ax1.set_xlabel('E (eV)')
    ax1.set_ylabel(r'$\log_{10}$($\psi(E)$) (arb. u.)')
    ax1.set_title('Time FFT')
    ax1.grid(axis='x')
    ie = np.searchsorted(telecx, elece)
    ax1.set_xlim(0, elece)
    ax1.set_ylim(min(np.log10(abs(telecy)**2)[:ie]), max(np.log10(abs(telecy)**2)[:ie]))

    lin2, = ax2.plot(selecx, np.log10(abs(selecy)**2), 'k-', lw=1)
    ax2.set_xlabel('E (eV)')
    ax2.set_ylabel(r'$\log_{10}$($\psi(E)$) (arb. u.)')
    ax2.set_title('Spatial FFT')
    ax2.grid(axis='x')
    ie = np.searchsorted(selecx, elece)
    ax2.set_xlim(0, elece)
    ax2.set_ylim(min(np.log10(abs(selecy)**2)[0:ie]), max(np.log10(abs(selecy)**2)[0:ie]))

    selmul = Slider(ax4, 'Space FFT Detection Zone', X[0], data['vdpos'], valinit=zim)
    
    
    elecx = telecx
    elecy = abs(telecy + np.interp(elecx, selecx, selecy))**2
    elecy = np.log10(elecy)
    lin3, = ax3.plot(elecx, elecy, 'k-', lw=1)
    ax3.set_title('Combined Spectrum')
    ax3.set_xlabel('E (eV)')
    ax3.set_ylabel(r'$\log_{10}(P(E))$ (arb. u.)')
    ax3.grid(axis='x')
    ax3.set_xlim(0, elece)
    

    def update(val):
        zimind = int(np.searchsorted(X, val))
        telecx, telecy = getFFT(tfunc, (ts[-1]-ts[0])/len(ts), tukeyFlatWindow(len(tfunc), True))
        sfunc = data['finalfunc'][zimind:detind]
        selecx, selecy = getFFT(sfunc, (X[-1]-X[0])/len(X), tukeyFlatWindow(len(sfunc), True, 1))
    
        telecx = telecx[0:int(len(telecx)/2)]*4.13567e-15
        telecy = telecy[0:int(len(telecy)/2)]/np.sqrt(hbar)

        selecx = (selecx[0:int(len(selecx)/2)]*hbar*2*np.pi)**2/(2*me*qe)
        selecy = selecy[0:int(len(selecy)/2)]*np.sqrt(me)/hbar
    
        elecx = telecx
        elecy = abs(telecy + np.interp(elecx, selecx, selecy))**2
        elecy = np.log10(elecy)
        lin3.set_data(elecx, elecy)
        ie = np.searchsorted(elecx, elece)
        ax3.set_ylim(min(elecy[0:ie]), max(elecy[0:ie]))
        selecy = np.log10(abs(selecy)**2)
        lin2.set_data(selecx, selecy)
        ie = np.searchsorted(selecx, elece)
        ax2.set_ylim(min(selecy[0:ie]), max(selecy[0:ie]))
        fig.canvas.draw_idle()
    selmul.on_changed(update)

    fig.suptitle(r'Electron Spectrum: $\lambda=$%d nm, $E_{max}=$%d GVm$^{-1}$, $\tau=$%d fs'%(data['lam']*1e9, data['Emax']*1e-9, data['tau']*1e15))

    plt.tight_layout()
    #plt.savefig(sfil.name, dpi=500)
    plt.show()

def saveElectronData(elece=2000):
    """Saves electron spectrum data to a file.
    Will ask to provide a data file(s).
    
    Args:
        elece (float, optional): Maximum energy to save in file, eV. This should correspond to about 10 Up. Defaults to 2000 eV.
    """
    
    root = tk.Tk()
    root.withdraw()
    fils = filedialog.askopenfiles()
    for fil in fils:
        data = np.load(fil.name)
        funcs = np.conj(data['sampfunc'])
        ts = data['ts']
        elecx, elecy = getFFT(funcs, (ts[-1]-ts[0])/len(ts), signal.hanning(len(funcs)))
        elecx = elecx[0:int(len(elecx)/2)]*4.13567e-15
        elecy = elecy[0:int(len(elecy)/2)]/np.sqrt(hbar)
        ie = np.searchsorted(elecx, elece)
        elecx = elecx[:ie]
        elecy = abs(elecy[:ie])**2
        np.savetxt(os.path.splitext(fil.name)[0]+".txt", np.transpose([elecx, elecy]), fmt = "%0.10E %0.10E", header='E(eV) |Psi(E)|^2', comments='')
        #saveName = os.path.splitext(fil.name)[0]+desc+".png"

def main():
    """Used to manually run a dataset."""
    plot = False
    Emax = 40e9
    lam = 800
    Xmax = 15e-9
    Xmin = -15e-9
    err = 1
    tprec = 1
    dx = getIdealDX(Emax, Xmax, err)
    X = np.arange(Xmin, Xmax, dx)
    X = X-X[len(X)//2]
    print("dx=%g"%dx)
    fname="%sla%dxma%dxmi%dem%der%dtp%d"%("",lam, Xmax*1e9, Xmin*1e9, Emax*1e-9, err*100, tprec*100)
    fol = "test"
    print(fname)
    tau = 40e-15
    generateData(X, Emax, lam*1e-9, 3e-9, tau=tau, totTime=400e-15, tprec=tprec, phase=3*np.pi/2, centMul=6, typ=2, bufferLeft=True, plotSaves=plot, itsBetweenSave=100, savePsi=True, savePsiRes=1000, saveWhenFinished= not plot, stopWhenFinished= not plot, printProgress=True, subFolder=fol, fname=fname)

def testGaussian(energ = 10):
    """Used to manually run a dataset.
    Args:
        energ (float, optional): Energy of gaussian wave. Defaults to 10 eV.
    """
    plot=False
    L = 1e-9
    Xmax = 40
    Xmin = -40
    Emax = 2e10
    err = 0.1
    tprec = 1
    lam=1800
    tau = 5e-15
    intspace = 100e-10
    tmax = getIdealMaxT(intspace, energ*eV_J/10)
    Xmax = intspace+getTravelDist(energ*eV_J*10, tmax)
    dx = getIdealDX(energ*eV_J/qe, 1, err)
    X = np.arange(Xmin*L, Xmax, dx)
    k = np.sqrt(2*me*energ*eV_J)/hbar
    psi = np.exp(-X**2*k**2/10000)*np.exp(1j*X*k)
    print("dx=%g"%dx)
    fname="elec%d"%(energ)
    fol = "testgauss"
    print(fname)
    generateData(X, Emax, lam*1e-9, 3*L, tprec=tprec, getGroundState=False, detPos=intspace, totTime=tmax, initialState=psi, phase=3*np.pi/2, centMul=4, typ=7, bufferLeft=True, plotSaves=plot, itsBetweenSave=100, savePsi=True, savePsiRes=1000, saveWhenFinished= not plot, stopWhenFinished= True, printProgress=True, subFolder=fol, fname=fname)

def plotImagTimeProp():
    """Saves a bunch of images which can be combined to animate imaginary time propagation."""
    root = tk.Tk()
    root.withdraw()
    fol = filedialog.askdirectory()
    L = 1e-9
    X = np.linspace(-3*L, 3*L, 1000)
    ipot = CompositePotentialFunction(X, [WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J), ShieldedPotential(X, 0, 2.5e-10, 1.74, 1e-10)])
    lf = ConstantField(X, 0, Envelope(), -3*L, 3*L)
    pot = PotentialFunction(X, ipot, lf)
    waveFunc = WaveFunction(pot, 0, False)
    for I in range(20):
        fig, ax1 = plt.subplots(1,1)
        ax1.plot(X*1e9, waveFunc.prob()*1e-10, 'b-')
        ax1.set_ylim(0, 2)
        ax1.set_xlabel('X (nm)')
        ax1.set_ylabel(r'$|\psi|^2$ ($\AA$$^{-1}$)')
        ax2 = plt.twinx()
        ax2.plot(X*1e9, ipot.getV()/eV_J, 'k-')
        ax2.set_ylabel(r'$V$ (eV)')
        ax1.set_xlim(X[0]*1e9, X[-1]*1e9)
        ax1.set_title('Imaginary Time Propagation')
        plt.savefig(fol + "/%d.png"%I)
        plt.close()
        waveFunc.findGroundState(1000)
        
def plotTimeProp():
    """Saves a bunch of images which can be combined to animate a simulation."""
    root = tk.Tk()
    root.withdraw()
    fol = filedialog.askdirectory()
    L = 1e-9
    X = np.linspace(-3*L, 10*L, 1000)
    ipot = SquareWell(X, 1*eV_J, L)
    lf = LightField(X, 5e9, 800e-9, np.pi/2, Envelope(), L*1.05, 10*L, 0)
    pot = PotentialFunction(X, ipot, lf)
    waveFunc = WaveFunction(pot, 2*L, True)
    waveFunc.findGroundState(20000)
    for I in range(50):
        fig, ax1 = plt.subplots(1,1)
        ax1.plot(X*1e9, waveFunc.prob()*1e-9, 'b-')
        ax1.set_ylim(0, 2)
        ax1.set_xlabel('X (nm)')
        ax1.set_ylabel(r'$|\psi|^2$ (nm$^{-1}$)')
        ax2 = plt.twinx()
        ax2.plot(X*1e9, pot.getV(waveFunc.rcurT)/eV_J, 'k-')
        ax2.set_ylabel(r'$V$ (eV)')
        ax2.set_ylim(-1.1, 0.1)
        ax1.set_xlim(X[0]*1e9, X[-1]*1e9)
        ax1.set_title('Real Time Propagation')
        plt.savefig(fol + "/%d.png"%I)
        plt.close()
        del fig, ax1, ax2
        for i in range(300):
            waveFunc.stepCN()

####################################################
#misc code for some plots           
#fig, ax1 = plt.subplots(1,1)
#L = 1e-9
#X = np.linspace(-3*L, 3*L, 1000)
#softatom = WachterAtomPotential(X, 9.2*eV_J)
#shield = ShieldedPotential(X, 0, 2.5e-10, 1.74, 1e-10)
#unshield = UnshieldedPotential(X, 0, 2.5e-10, 1.3)
#ax1.plot(X*1e9, softatom.getV()/eV_J, 'k-', label='Wachter Atom, $E_f$ = 9.2 eV')
#ax1.plot(X*1e9, shield.getV()/eV_J, 'b-', label='Shielded Atom, $d$ = 2.5 $\AA$, $\kappa$ = 10 nm$^{-1}$, $Z$ = 1.74')
#ax1.plot(X*1e9, unshield.getV()/eV_J, 'r-', label='Unshielded Atom, $d$ = 2.5 $\AA$, $Z$ = 1.3')
#ax1.set_xlabel('X (nm)')
#ax1.set_ylabel('$V$ (eV)')
#ax1.set_title('Atomic Potentials')
#ax1.legend()
#plt.show()

#fig, ax1 = plt.subplots(1,1)
#L = 1e-9
#X = np.linspace(-3*L, 3*L, 1000)
#ipot = CompositePotentialFunction(X, [WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J), ShieldedPotential(X, 0, 2.5e-10, 1.74, 1e-10)])
#ax1.plot(X*1e9, ipot.getV()/eV_J, 'k-')
#ax1.plot(np.array([1,1])*WachterJelliumPotential(X, 9.2*eV_J, 6.2*eV_J).zim*1e9, [0, -35], 'k--')
#ax1.set_xlabel('X (nm)')
#ax1.set_ylabel('V (eV)')
#ax1.set_title('Jellium and Shielded Atom Potentials')
#plt.show()

###################################################
#Uses parallel processing to generate a data set. Set the settings in the code after worker, and uncomment
#L = 1e-9
#Xmax = 40
#Xmin = -15
#Emax = 2e10
#err = 0.2
#tprec = 2
#lam=2000
#dx = getIdealDX(Emax, Xmax, L, err)
#X = np.arange(Xmin, Xmax, dx)*L
#    
#def worker(args):
#    Emax = args[0]*1e9
#    lam = args[1]
##     acc = arg   
##    if typ == 0:
##        fol = "10umwachter"
##    elif typ == 1:
##        fol = "10umunshielded"
##    elif typ == 2:
##        fol = "10umshielded"
##    else:
##        fol = "10umsoftshielded"
#    fol = "erlangen"  
#    Xmax = 40
#    dx = getIdealDX(Emax, Xmax, L, err)
#    X = np.arange(Xmin, Xmax, dx)*L
#    
#    fname="%sla%dxma%dxmi%dem%der%dtp%d"%("",lam, Xmax, Xmin, Emax*1e-8, err*100, tprec*100)
#    print(fname)
#    generateData(X, Emax, lam*1e-9, 3*L, tprec=tprec, phase=3*np.pi/2, typ=2, bufferLeft=True, plotSaves=False, itsBetweenSave=500, oscToSave=5, savePsi=True, savePsiRes=1000, saveWhenFinished= True, stopWhenFinished= True, printProgress=False, subFolder=fol, fname=fname)
#
#if __name__=='__main__':
#    pool = Pool(processes=7) #SET NUMBER OF PROCESSES
#    
#    ins = []
#    for i in [20, 18.5, 16.9, 15.2, 13.2, 10.8, 7.8]: #SET E-FIELDS (GV/m)
#       for j in [1880]: #SET WAVELENGTHS (nm)
#           ins.append([i,j])
#        
#    pool.map(worker, ins)
#
#import profile
#profile.run("main()")