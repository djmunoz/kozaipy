import numpy as np
import scipy.integrate as integ
import kozaipy.triples as triples
from .triples_integrate_full import *
from .triples_integrate_tides import *
from .triples_integrate_single_average import *
#import bsint


triple_precision = {"e1x": 1.0e-8,
                    "e1y": 1.0e-8,
                    "e1z": 1.0e-8,
                    "l1x": 1.0e-10,
                    "l1y": 1.0e-10,
                    "l1z": 1.0e-10,
                    "e2x": 1.0e-6,
                    "e2y": 1.0e-6,
                    "e2z": 1.0e-6,
                    "l2x": 1.0e-8,
                    "l2y": 1.0e-8,
                    "l2z": 1.0e-8,
                    "spin0x": 1.0e-8,
                    "spin0y": 1.0e-8,
                    "spin0z": 1.0e-8,
                    "spin1x": 1.0e-7,
                    "spin1y": 1.0e-7,
                    "spin1z": 1.0e-7,
                    "Omega0": 1.0e-7,
                    "Omega0x": 1.0e-7,
                    "Omega0y": 1.0e-7,
                    "Omega0z": 1.0e-7,
                    "Omega1": 1.0e-5,
                    "Omega1x": 1.0e-5,
                    "Omega1y": 1.0e-5,
                    "Omega1z": 1.0e-5,
                    "m0": 1e-8,
                    "m1": 1e-8,
                    "m2": 1e-8,
                    "R0": 1e-10,
                    "R1": 1e-10}

def integrate_triple_system(ics,timemin,timemax,Nevals,
                            body0, body1, body2,
                            octupole_potential = True,
                            short_range_forces_conservative= False,
                            short_range_forces_dissipative = False,
                            solve_for_spin_vector=False,
                            version = 'tides',
                            tol = 1.0e-10,
                            integrator='scipy',
                            add_derivatives = False):


    atol = tol  
    rtol = tol/10.0
    rtol = 1.0e-8


    m0,m1,m2 = body0.mass, body1.mass, body2.mass
    radius0, radius1 = body0.radius, body1.radius
    dradius0_dt, dradius1_dt = body0.dradius_dt, body1.dradius_dt
    gyroradius0, gyroradius1 = body0.gyroradius, body1.gyroradius
    dgyroradius0_dt, dgyroradius1_dt = body0.dgyroradius_dt, body1.dgyroradius_dt
    k2_0, k2_1 = body0.apsidal_constant, body1.apsidal_constant
    tv0, tv1 = body0.viscous_time, body1.viscous_time
    tauconv0, tauconv1 = body0.convective_time, body1.convective_time
    tlag0, tlag1 = body0.tidal_lag_time, body1.tidal_lag_time

    rtol,atol = np.zeros(len(ics)),np.zeros(len(ics))
    for key,val in triples.triple_keys.items():
        if val is not None:
            rtol[val] = 10*triple_precision[key]
            atol[val] = triple_precision[key]

    #rtol,atol = 1.e-9,1.e-08
            
    #integrator='other'
    #integrator = 'bsint'
    
    time = np.linspace(timemin,timemax,Nevals)
    params = m0,m1,m2,radius0,radius1,gyroradius0,gyroradius1,k2_0,k2_1,\
             tv0,tv1,tauconv0,tauconv1,\
             tlag0,tlag1,\
             dradius0_dt, dradius1_dt,\
             dgyroradius0_dt, dgyroradius1_dt,\
             octupole_potential,\
             short_range_forces_conservative,short_range_forces_dissipative,solve_for_spin_vector    

    if (integrator == 'scipy'):
        if (version == 'tides'):

            params = m0,m1,m2,radius0,radius1,gyroradius0,gyroradius1,k2_0,k2_1,tv0,tv1,\
                     octupole_potential,\
                     short_range_forces_conservative,short_range_forces_dissipative,solve_for_spin_vector    

            sol =integ.odeint(threebody_ode_vf_tides_modified,ics,time,\
                              args=params,\
                              atol=atol,rtol=rtol,mxstep=1000000,hmin=0.0000001,mxords=16,mxordn=12)
        elif (version == 'full'):    
            sol =integ.odeint(threebody_ode_vf_full_modified,ics,time,\
                              args=params,\
                              atol=atol,rtol=rtol,mxstep=1000000,hmin=0.000000001,mxords=12,mxordn=10)
            if (add_derivatives):
                for kk,y in enumerate(sol):
                    if (kk == 0):
                        dsol_dt = threebody_ode_vf_full_modified(y,time[kk],*params)[:len(triples.triple_derivative_keys)]
                    else:
                        dsol_dt = np.vstack((dsol_dt,threebody_ode_vf_full_modified(y,time[kk],*params)[:len(triples.triple_derivative_keys)]))


    else:
        if (version == 'tides'):
            params = [p for p in params if p is not None]
            solver = integ.ode(threebody_ode_vf_tides).set_integrator('dopri',nsteps=3000000,atol=atol,rtol=rtol, method='bdf')
        elif (version == 'full'):
            #solver = integ.ode(threebody_ode_vf_full).set_integrator('lsoda',nsteps=3000000,atol=atol,rtol=rtol, method='bdf',min_step=0.00001,max_order_ns=10)
            solver = integ.ode(threebody_ode_vf_full).set_integrator('dopri',nsteps=3000000,atol=atol,rtol=rtol, method='bdf')

        solver.set_initial_value(ics, time[0]).set_f_params(*params)
        kk = 1
        sol = []
        sol.append(ics)
        while solver.successful() and solver.t < time[-1]:
            solver.integrate(time[kk])
            sol.append(solver.y)
            kk+=1
        sol = np.asarray(sol)


    retval = np.column_stack((time,sol))

    if add_derivatives: retval = retval, dsol_dt
    
    return retval





def integrate_triple_system_sa(ics,timemin,timemax,Nevals,
                               body0, body1, body2,
                               octupole_potential = True,
                               short_range_forces_conservative= False,
                               short_range_forces_dissipative = False,
                               solve_for_spin_vector=False,
                               fix_outer_orbit = True,
                               tol = 1.0e-10):


    if (fix_outer_orbit):
        # turn off integration of external orbit
        triples.triple_keys['e2x'], triples.triple_keys['e2y'], triples.triple_keys['e2z'] =  None, None, None
        triples.triple_keys['l2x'], triples.triple_keys['l2y'], triples.triple_keys['l2z'] =  None, None, None

    m0,m1,m2 = body0.mass, body1.mass, body2.mass
    radius0, radius1 = body0.radius, body1.radius
    dradius0_dt, dradius1_dt = body0.dradius_dt, body1.dradius_dt
    gyroradius0, gyroradius1 = body0.gyroradius, body1.gyroradius
    dgyroradius0_dt, dgyroradius1_dt = body0.dgyroradius_dt, body1.dgyroradius_dt
    k2_0, k2_1 = body0.apsidal_constant, body1.apsidal_constant
    tv0, tv1 = body0.viscous_time, body1.viscous_time
    tauconv0, tauconv1 = body0.convective_time, body1.convective_time
    tlag0, tlag1 = body0.tidal_lag_time, body1.tidal_lag_time

    rtol,atol = np.zeros(len(ics)),np.zeros(len(ics))
    for key,val in triples.triple_keys.items():
        if val is not None:
            rtol[val] = 10*triple_precision[key]
            atol[val] = triple_precision[key]
    
    params = m0,m1,m2,radius0,radius1,gyroradius0,gyroradius1,k2_0,k2_1,\
             tv0,tv1,tauconv0,tauconv1,\
             tlag0,tlag1,\
             dradius0_dt, dradius1_dt,\
             dgyroradius0_dt, dgyroradius1_dt,\
             octupole_potential,\
             short_range_forces_conservative,short_range_forces_dissipative,solve_for_spin_vector    


    
    if (fix_outer_orbit):
        new_params = ics[6:12]
        ics = np.append(ics[0:6],ics[12:])
        rtol = np.append(rtol[0:6],rtol[12:])
        atol = np.append(atol[0:6],atol[12:])
        for p in params: new_params.append(p)
        params = tuple(new_params)




        
    time = np.linspace(timemin,timemax,Nevals)


    sol =integ.odeint(threebody_ode_vf_sa,ics,time,\
                      args=params,\
                      mxstep=1000000,hmin=0.000000001,mxords=12,mxordn=10)
                      #atol=atol,rtol=rtol)#,

    
    retval = np.column_stack((time,sol))

    return retval
