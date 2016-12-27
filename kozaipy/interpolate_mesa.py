import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from string import split
from triples import *
radius_file = './R.txt'
gyroradius_file = './Rg.txt'
time_file = './t.txt'

planet_file = './hd209458b/cold-diss-1e-5/evol.des'

lagtime_file = './time_lag.txt'

time_throwout = 1e3*365.25

def interpolate_radius():

    radius = np.loadtxt(radius_file)
    time = np.loadtxt(time_file) * 365.25
    radius = radius[time > time_throwout]
    time = time[time > time_throwout]# - time_throwout
    
    dt = time[-1] - time[-2]
    radius = np.append(radius,np.repeat(radius[-1],int(time[-1]/dt)))
    time=np.append(time,np.linspace(dt+time[-1],2*time[-1],int(time[-1]/dt)))
    dradius_dtime = np.gradient(radius)/np.gradient(time)

    radius_func = interp1d(time,radius,fill_value='extrapolate')
    dradius_dt_func = interp1d(time,dradius_dtime,fill_value='extrapolate')

    
    return radius_func, dradius_dt_func

def interpolate_gyroradius():

    gyroradius = np.loadtxt(gyroradius_file)
    time = np.loadtxt(time_file) * 365.25
    gyroradius = gyroradius[time > time_throwout]
    time = time[time > time_throwout]
    
    dt = time[-1] - time[-2]
    gyroradius = np.append(gyroradius,np.repeat(gyroradius[-1],int(time[-1]/dt)))
    time=np.append(time,np.linspace(dt+time[-1],2*time[-1],int(time[-1]/dt)))
    dgyroradius_dtime = np.gradient(gyroradius)/np.gradient(time)

    gyroradius_func = interp1d(time,gyroradius,fill_value='extrapolate')
    dgyroradius_dt_func = interp1d(time,dgyroradius_dtime,fill_value='extrapolate')

    
    return gyroradius_func, dgyroradius_dt_func


def interpolate_planet_radius():

    with open(planet_file) as f:
        nlines=0
        nread=0
        t, R = [],[]
        for line in f:
            if ("C" in line): continue
            if (nlines == 0):
                nread = int(line)
                nlines+=1
                continue
            data = split(line)
            t.append(float(data[0]))
            R.append(float(data[1]))
            nlines+=1
            if (nlines == nread): break
    
    radius = np.asarray(R) * constants.Rjup
    time = np.asarray(t) * 1e6 * 365.25
    radius = radius[time > time_throwout]
    time = time[time > time_throwout]# - time_throwout
    
    dt = time[-1] - time[-2]
    radius = np.append(radius,np.repeat(radius[-1],int(time[-1]/dt)))
    time=np.append(time,np.linspace(dt+time[-1],2*time[-1],int(time[-1]/dt)))
    dradius_dtime = np.gradient(radius)/np.gradient(time)

    radius_func = interp1d(time,radius,fill_value='extrapolate')
    dradius_dt_func = interp1d(time,dradius_dtime,fill_value='extrapolate')

    return radius_func, dradius_dt_func


def interpolate_lagtime():

    data = np.loadtxt(lagtime_file)
    time = data[:,0]*365.25
    lag = data[:,1]*365.25#/20

    lag = lag[time > time_throwout]
    time = time[time > time_throwout]

    dt = time[-1] - time[-2]
    lag = np.append(lag,np.repeat(lag[-1],int(time[-1]/dt)))
    time=np.append(time,np.linspace(dt+time[-1],2*time[-1],int(time[-1]/dt)))
    dlag_dtime = np.gradient(lag)/np.gradient(time)

    lag_func = interp1d(time,lag,fill_value='extrapolate')
    dlag_dt_func = interp1d(time,dlag_dtime,fill_value='extrapolate')

    
    
    return lag_func, dlag_dt_func
