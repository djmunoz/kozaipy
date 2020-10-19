from math import sqrt, pi,log10, cos, sin
from numpy import inf
import scipy.integrate as integ
from .triples import *
#import bsint

def threebody_ode_vf_sa(y,t,
                        eoutx,eouty,eoutz,houtx,houty,houtz,
                        m0,m1,m2,
                        radius0, radius1,
                        gyroradius0, gyroradius1,
                        k2_0, k2_1,
                        tv0, tv1,
                        tauconv0, tauconv1,
                        tlag0, tlag1,
                        dradius0_dt,dradius1_dt,
                        dgyroradius0_dt, dgyroradius1_dt,
                        octupole,
                        extra_forces_conservative,
                        extra_forces_dissipative,
                        solve_for_spin_vector):


    #####################################################################
    #time-dependent variables ###########################################
    
    
    # get the active variables
    jj = 0
    # for the inner binary
    if (triple_data['inner_orbit']):
        einx = y[jj+0]
        einy = y[jj+1]
        einz = y[jj+2]
        hinx = y[jj+3]
        hiny = y[jj+4]
        hinz = y[jj+5]
        jj+=6


    # for the spin (specific) angular momenta
    if (solve_for_spin_vector):
        if (triple_data['spin0']):
            if not  (triple_data['spinorbit_align0']):
                Spin0x = y[jj+0]
                Spin0y = y[jj+1]
                Spin0z = y[jj+2]
                jj+=3
            else:
                Spin0 = y[jj+0]
                jj+=1
        if (triple_data['spin1']):
            if not  (triple_data['spinorbit_align1']):
                Spin1x = y[jj+0]
                Spin1y = y[jj+1]
                Spin1z = y[jj+2]
                jj+=3
            else:
                Spin1 = y[jj+0]
                jj+=1              
    else:
        if (triple_data['spin0']):
            if not  (triple_data['spinorbit_align0']):
                Omega0x = y[jj+0]
                Omega0y = y[jj+1]
                Omega0z = y[jj+2]
                jj+=3
            else:
                Omega0 = y[jj+0]
                jj+=1
        if (triple_data['spin1']):
            if not  (triple_data['spinorbit_align1']):
                Omega1x = y[jj+0]
                Omega1y = y[jj+1]
                Omega1z = y[jj+2]
                jj+=3      
            else:
                Omega1 = y[jj+0]
                jj+=1 
    # radii
    if (radius0 is not None):
        if callable(radius0):
            R0 = radius0(t)
        else:
            R0 = radius0

    if (radius1 is not None):
        if callable(radius1):
            R1 = radius1(t)
        else:
            R1 = radius1


    if (dradius0_dt is not None):
        if callable(dradius0_dt):
            dR0_dt = dradius0_dt(t)
        else:
            dR0_dt = dradius0_dt
            
            
    if (dradius1_dt is not None):
        if callable(dradius1_dt):
            dR1_dt = dradius1_dt(t)
        else:
            dR1_dt = dradius1_dt

    # gyroradii
    if (gyroradius0 is not None):
        if callable(gyroradius0):
            rg_0 = gyroradius0(t)
            # hot fix
            k2_0 = 10**((log10(rg_0)+0.13)/0.24)
        else:
            rg_0 = gyroradius0

    if (gyroradius1 is not None):
        if callable(gyroradius1):
            rg_1 = gyroradius1(t)
        else:
            rg_1 = gyroradius1


    if (dgyroradius0_dt is not None):
        if callable(dgyroradius0_dt):
            drg0_dt = dgyroradius0_dt(t)
        else:
            drg0_dt = dgyroradius0_dt
            
            
    if (dgyroradius1_dt is not None):
        if callable(dgyroradius1_dt):
            drg1_dt = dgyroradius1_dt(t)
        else:
            drg1_dt = dgyroradius1_dt

    # time lags
    if (tlag0 is not None):
        if callable(tlag0):
            taulag0 = tlag0(t)
        else:
            taulag0 = tlag0

    if (tlag1 is not None):
        if callable(tlag1):
            taulag1 = tlag1(t)
        else:
            taulag1 = tlag1

    ####################################################
    # Some quantities defined for convenience
    mu = m0 * m1 / (m0 + m1)
    
    ein_squared = einx * einx + einy * einy + einz * einz
    ein = sqrt(ein_squared)
    one_minus_einsq = 1 - ein_squared
    one_minus_einsq_sqrt = sqrt(one_minus_einsq)
    one_minus_einsq_squared = one_minus_einsq * one_minus_einsq
    one_minus_einsq_fifth =  one_minus_einsq_squared * one_minus_einsq_squared * one_minus_einsq
    hin_squared = hinx * hinx + hiny * hiny + hinz * hinz
    hin = sqrt(hin_squared)
    
    eout_squared = eoutx * eoutx + eouty * eouty + eoutz * eoutz
    hout = sqrt(houtx * houtx + houty * houty + houtz * houtz)
    eout = sqrt(eout_squared)

    
    Gm_in =  constants.G * (m0 + m1)
    Gm_out =  constants.G * (m0 + m1 + m2)
    ain = hin_squared / (1 - ein * ein) / Gm_in
    aout = hout * hout / (1 - eout * eout) / Gm_out
    norbit_in = sqrt(Gm_in/ain/ain/ain)
    norbit_out = sqrt(Gm_out/aout/aout/aout)
    L_in = sqrt(Gm_in * ain)
    L_out = sqrt(Gm_out * aout)

    # unit vectors
    uinx, uiny, uinz = einx/ein, einy/ein, einz/ein
    ninx, niny, ninz = hinx/hin, hiny/hin, hinz/hin
    vinx = (niny * uinz - ninz * uiny)
    viny = (ninz * uinx - ninx * uinz)
    vinz = (ninx * uiny - niny * uinx)
    noutx, nouty, noutz = houtx/hout, houty/hout, houtz/hout
    uoutx, uouty, uoutz = eoutx/eout, eouty/eout, eoutz/eout
    voutx = (nouty * uoutz - noutz * uouty)
    vouty = (noutz * uoutx - noutx * uoutz)
    voutz = (noutx * uouty - nouty * uoutx)

    nindotnout =  noutx * ninx + nouty * niny + noutz * ninz
    uindotnout =  noutx * uinx + nouty * uiny + noutz * uinz
    uindotuout =  uoutx * uinx + uouty * uiny + uoutz * uinz
    nindotuout =  uoutx * ninx + uouty * niny + uoutz * ninz
    uindotvout =  voutx * uinx + vouty * uiny + voutz * uinz
    nindotvout =  voutx * ninx + vouty * niny + voutz * ninz
    nincrossuin_x = niny * uinz - ninz * uiny
    nincrossuin_y = ninz * uinx - ninx * uinz
    nincrossuin_z = ninx * uiny - niny * uinx
    nincrossuout_x = niny * uoutz - ninz * uouty
    nincrossuout_y = ninz * uoutx - ninx * uoutz
    nincrossuout_z = ninx * uouty - niny * uoutx
    nincrossnout_x = niny * noutz - ninz * nouty
    nincrossnout_y = ninz * noutx - ninx * noutz
    nincrossnout_z = ninx * nouty - niny * noutx 
    uincrossuout_x = uiny * uoutz - uinz * uouty
    uincrossuout_y = uinz * uoutx - uinx * uoutz
    uincrossuout_z = uinx * uouty - uiny * uoutx   
    uincrossnout_x = uiny * noutz - uinz * nouty
    uincrossnout_y = uinz * noutx - uinx * noutz
    uincrossnout_z = uinx * nouty - uiny * noutx   


    # Prescribed outer orbit using elliptic expansions
    cosMout = cos(norbit_out * t)
    cos2Mout = cos(2 * norbit_out * t)
    cos3Mout = cos(3 * norbit_out * t)
    cos4Mout = cos(4 * norbit_out * t)
    cos5Mout = cos(5 * norbit_out * t)
    sinMout = sin(norbit_out * t)
    sin2Mout = sin(2 * norbit_out * t)
    sin3Mout = sin(3 * norbit_out * t)
    sin4Mout = sin(4 * norbit_out * t)
    sin5Mout = sin(5 * norbit_out * t)

    routoveraout = 1 + eout * (-cosMout + eout * (0.5 - 0.5 * cos2Mout + eout * 0.375 * (cosMout - cos3Mout )))
    
    aoutoverrout = 1 + eout * (cosMout +  eout * (cos2Mout  + 0.125 * eout * \
                                                  (9 * cos3Mout - cosMout)))
    aoutoverrout3 = 1 + eout * (3 * cosMout) + eout**2 * (1.5 + 4.5 * cos2Mout) +\
        eout**3 * (3.375 * cosMout + 6.625 * cos3Mout)

    sinfout = sinMout + eout * sin2Mout + eout**2 * (9./8 * sin3Mout - 7./8 * sinMout) + eout**3 * (4./3 * sin4Mout - 7./6 * sin2Mout) +\
        eout**4 * (17./192 * sinMout - 207./128 * sin3Mout + 625./384 * sin5Mout)
                  
    cosfout = cosMout + eout * (cos2Mout - 1) + 9./8 * eout**2 * (cos3Mout - cosMout) + 4./3 * eout**3 * (cos4Mout - cos2Mout) + \
        eout**4 * (25./192 * cosMout - 225./128 * cos3Mout + 625./384 * cos5Mout)
    
    cos2f = cosfout * cosfout - sinfout * sinfout

    sin2f = 2 * sinfout * cosfout
    
    routx = cosfout * uoutx + sinfout * voutx
    routy = cosfout * uouty + sinfout * vouty
    routz = cosfout * uoutz + sinfout * voutz

    
    # Vector products
    uindotrout = uinx * routx + uiny * routy + uinz * routz
    nindotrout = ninx * routx + niny * routy + ninz * routz
    nincrossrout_x = niny * routz - ninz * routy
    nincrossrout_y = ninz * routx - ninx * routz
    nincrossrout_z = ninx * routy - niny * routx
    uincrossrout_x = uiny * routz - uinz * routy
    uincrossrout_y = uinz * routx - uinx * routz
    uincrossrout_z = uinx * routy - uiny * routx
    


    ein_fourth = ein_squared * ein_squared
    ein_sixth = ein_fourth * ein_squared
    f2 = 1 + 7.5 * ein_squared + 5.625 * ein_fourth + 0.3125 * ein_sixth
    f3 = 1 + 3.75 * ein_squared + 1.875 * ein_fourth + 0.078125 * ein_sixth
    f4 = 1 + 1.5 * ein_squared + 0.125 * ein_fourth
    f5 = 1 + 3.0 * ein_squared + 0.375 * ein_fourth

        
    if (triple_data['spin0']):
        I0 = rg_0 * m0 * R0 * R0
        if (dradius0_dt is not None):
            dI0_dt = 2 * rg_0 * m0 * R0 * dR0_dt
            if (dgyroradius0_dt is not None):
                dI0_dt+= drg0_dt * m0 * R0 * R0
        else:
            dI0_dt = 0
        if not  (triple_data['spinorbit_align0']):
            if (solve_for_spin_vector):
                Omega0_u = (Spin0x * uinx + Spin0y * uiny + Spin0z * uinz) / I0
                Omega0_n = (Spin0x * ninx + Spin0y * niny + Spin0z * ninz) / I0 
                Omega0_v = (Spin0x * vinx + Spin0y * viny + Spin0z * vinz) / I0
            else:
                Omega0_u = (Omega0x * uinx + Omega0y * uiny + Omega0z * uinz)
                Omega0_n = (Omega0x * ninx + Omega0y * niny + Omega0z * ninz)
                Omega0_v = (Omega0x * vinx + Omega0y * viny + Omega0z * vinz)            
            Omega0 = sqrt(Omega0_u**2 + Omega0_v**2 + Omega0_n**2)
        else:
            Omega0_u = 0
            Omega0_v = 0
            if (solve_for_spin_vector):
                Omega0_n = Spin0 / I0 
            else:
                Omega0_n = Omega0 
    elif (triple_data['pseudosynch0']):
        Omega0_u = 0
        Omega0_n = f2/f5/one_minus_einsq/one_minus_einsq_sqrt * norbit_in
        Omega0_v = 0
        Omega0 = Omega0_n
    else:
        Omega0_u, Omega0_v, Omega0_n, Omega0 = 0, 0, 0, 0

        
    if (triple_data['spin1']):
        I1 = rg_1 * m1 * R1 * R1
        if (dradius1_dt is not None):
            dI1_dt = 2 * rg_1 * m1 * R1 * dR1_dt
            if (dgyroradius1_dt is not None):
                 dI1_dt+= drg1_dt * m1 * R1 * R1
        else:
            dI1_dt = 0
        if not  (triple_data['spinorbit_align1']):
            if (solve_for_spin_vector):
                Omega1_u = (Spin1x * uinx + Spin1y * uiny + Spin1z * uinz) / I1 
                Omega1_n = (Spin1x * ninx + Spin1y * niny + Spin1z * ninz) / I1  
                Omega1_v = (Spin1x * vinx + Spin1y * viny + Spin1z * vinz) / I1
            else:
                Omega1_u = (Omega1x * uinx + Omega1y * uiny + Omega1z * uinz)
                Omega1_n = (Omega1x * ninx + Omega1y * niny + Omega1z * ninz)
                Omega1_v = (Omega1x * vinx + Omega1y * viny + Omega1z * vinz)
            Omega1 = sqrt(Omega1_u**2 + Omega1_v**2 + Omega1_n**2)
        else:
            Omega1_u = 0
            Omega1_v = 0
            if (solve_for_spin_vector):
                Omega1_n = Spin1 / I1 
            else:
                Omega1_n = Omega1
    elif (triple_data['pseudosynch1']):
        Omega1_u = 0
        Omega1_n = f2/f5/one_minus_einsq/one_minus_einsq_sqrt * norbit_in
        Omega1_v = 0
        Omega = Omega1_n
    else:
        Omega1_u, Omega1_v, Omega1_n, Omega1 = 0, 0, 0, 0



        
    if (extra_forces_conservative):
        
        size_ratio0 = R0/ain
        size_ratio0_fifth = size_ratio0 * size_ratio0 * size_ratio0 * size_ratio0 * size_ratio0
        size_ratio0_eighth = size_ratio0 * size_ratio0 * size_ratio0 * size_ratio0_fifth
        size_ratio1 = R1/ain
        size_ratio1_fifth = size_ratio1 * size_ratio1 * size_ratio1 * size_ratio1 * size_ratio1
        size_ratio1_eighth = size_ratio1 * size_ratio1 * size_ratio1 * size_ratio1_fifth

                
        V0 = 0
                      
        W0 = 0

        X0 = -1.0/norbit_in * m1 * k2_0 * size_ratio0_fifth / mu  *  Omega0_n *  Omega0_u / one_minus_einsq_squared
        
        Y0 = -1.0/norbit_in * m1 * k2_0 * size_ratio0_fifth / mu  *  Omega0_n *  Omega0_v / one_minus_einsq_squared
        
        Z0 = 1.0/norbit_in * m1 * k2_0 * size_ratio0_fifth /mu * (0.5 * (2 * Omega0_n**2 - Omega0_u**2 - Omega0_v**2) / one_minus_einsq_squared \
                                                                  + 15 * constants.G * m1 / ain**3 * f4 / one_minus_einsq_fifth)

        V1 = 0
                      
        W1 = 0
        
        X1 = -1.0/norbit_in * m0 * k2_1 * size_ratio1_fifth / mu  *  Omega1_n *  Omega1_u / one_minus_einsq_squared
                
        Y1 = -1.0/norbit_in * m0 * k2_1 * size_ratio1_fifth / mu  *  Omega1_n *  Omega1_v / one_minus_einsq_squared
        
        Z1 = 1.0/norbit_in * m0 * k2_1 * size_ratio1_fifth /mu * (0.5*(2 * Omega1_n**2 - Omega1_u**2 - Omega1_v**2) / one_minus_einsq_squared \
                                                                  + 15 * constants.G * m0 / ain**3 * f4 / one_minus_einsq_fifth)
        
        
        ZGR = 3 * constants.G * (m0 + m1) * norbit_in / ain / constants.CLIGHT / constants.CLIGHT / one_minus_einsq
    

        if (extra_forces_dissipative):

            #tf0 = tv0/9 / size_ratio0_eighth * m0**2 / ((m0 + m1)*m1) / (1 + 2 * k2_0)**2 
            #tf1 = tv1/9 / size_ratio1_eighth * m1**2 / ((m0 + m1)*m0) / (1 + 2 * k2_1)**2 

            
            if (tv0 is not None):
                timelag0 = 1.5 / tv0 * R0 * R0 * R0 / constants.G / m0 * (1 + 2 * k2_0)**2/ k2_0
            elif (tauconv0 is not None):
                #ptide0 = 2*pi/np.abs(norbit_in - Omega0)
                #fconv0 = min(1.0, (ptide0/tauconv0/2) * (ptide0/tauconv0/2))
                fconv0 = 1
                timelag0 = 2.0/21 * fconv0/tauconv0 * R0 * R0 * R0 / constants.G / m0 / k2_0
            elif (tlag0 is not None):
                timelag0 = taulag0
            else:
                timelag0 = 0    

            if (tv1 is not None):
                timelag1 = 1.5 / tv1 * R1 * R1 * R1 / constants.G / m1 * (1 + 2 * k2_1)**2/ k2_1
            elif (tauconv1 is not None):
                #ptide1 = 2*pi/np.abs(norbit_in - Omega1)
                #fconv1 = min(1.0, (ptide1/tauconv1/2) * (ptide1/tauconv1/2))
                fconv1 = 1
                timelag1 = 2.0/21 * fconv1/tauconv1 * R1 * R1 * R1 / constants.G / m1 / k2_1
            elif (tlag1 is not None):
                timelag1 = taulag1
            else:
                timelag1 = 0   

            tf0 = m0 /m1 / size_ratio0_fifth / norbit_in /norbit_in / timelag0 /6 / k2_0
            tf1 = m1 /m0 / size_ratio1_fifth / norbit_in /norbit_in / timelag1 /6 / k2_1

            
            V0 += 9.0 / tf0 * (f3 / one_minus_einsq_fifth/one_minus_einsq/one_minus_einsq_sqrt - \
                               11.0/18 * Omega0_n / norbit_in * f4 / one_minus_einsq_fifth)

            W0 += 1.0 / tf0 * (f2 / one_minus_einsq_fifth/one_minus_einsq/one_minus_einsq_sqrt - \
                               Omega0_n / norbit_in * f5 / one_minus_einsq_fifth)    
            
            X0 += -1.0/norbit_in * Omega0_v / 2 / tf0 * (1 + 4.5 * ein_squared + 0.625 * ein_fourth) / one_minus_einsq_fifth
        
            Y0 += 1.0/norbit_in * Omega0_u / 2 / tf0 * (1 + 1.5 * ein_squared + 0.125 * ein_fourth) / one_minus_einsq_fifth
            

        
            V1 += 9.0 / tf1 * (f3 / one_minus_einsq_fifth/one_minus_einsq/one_minus_einsq_sqrt - \
                               11.0/18 * Omega1_n /norbit_in * f4 / one_minus_einsq_fifth)
            
            W1 += 1.0 / tf1 * (f2 / one_minus_einsq_fifth/one_minus_einsq/one_minus_einsq_sqrt - \
                               Omega1_n / norbit_in * f5 /one_minus_einsq_fifth)    
            
            X1 += -1.0/norbit_in * Omega1_v / 2 / tf1 * (1 + 4.5 * ein_squared + 0.625 * ein_fourth) / one_minus_einsq_fifth
            
            Y1 += 1.0/norbit_in * Omega1_u / 2 / tf1 * (1 + 1.5 * ein_squared + 0.125 * ein_fourth) / one_minus_einsq_fifth

    else:
        V0, W0, X0, Y0, Z0 = 0, 0, 0, 0, 0
        V1, W1, X1, Y1, Z1 = 0, 0, 0, 0, 0
        ZGR = 0

    ##########################################################################
    # System of differential equations


    # Prescribed outer orbit using elliptic expansions
    cosMout = cos(norbit_out * t)
    cos2Mout = cos(2 * norbit_out * t)
    cos3Mout = cos(3 * norbit_out * t)
    cos4Mout = cos(4 * norbit_out * t)
    sinMout = sin(norbit_out * t)
    sin2Mout = sin(2 * norbit_out * t)
    sin3Mout = sin(3 * norbit_out * t)
    sin4Mout = sin(4 * norbit_out * t)
    
    aoutoverrout = 1 + eout * (cosMout +  eout * (cos2Mout  + 0.125 * eout * \
                                                  (9 * cos3Mout - cosMout)))
    aoutoverrout3 = 1 + eout * (3 * cosMout + eout * (1.5 + 4.5 * cos2Mout +
                                                      eout * (3.375 * cosMout + 6.625 * cos3Mout)))

    sinfout = sinMout + eout * (sin2Mout + eout * (1.125 * sin3Mout - 0.875 * sinMout + eout *
                                                   (1.333333333333333 * sin4Mout - 1.16666666667 * sin2Mout)))
    cosfout = cosMout + eout * (cos2Mout - 1 + eout * (1.125 * cos3Mout - 1.125 * cosMout + eout *  1.333333333333333 *
                                                       (cos4Mout - cos2Mout)))

    routx = cosfout * uoutx + sinfout * voutx
    routy = cosfout * uouty + sinfout * vouty
    routz = cosfout * uoutz + sinfout * voutz
    # Vector products
    uindotrout = cosfout * uindotuout + sinfout * uindotvout
    nindotrout = cosfout * nindotuout + sinfout * nindotvout
    nincrossrout_x = niny * routz - ninz * routy
    nincrossrout_y = ninz * routx - ninx * routz
    nincrossrout_z = ninx * routy - niny * routx
    uincrossrout_x = uiny * routz - uinz * routy
    uincrossrout_y = uinz * routx - uinx * routz
    uincrossrout_z = uinx * routy - uiny * routx

    # Equations of motion at the quadrupole level:
    
    epsilon_out = m2 / (m0 + m1) * ain * ain * ain / aout / aout / aout


    # For the inner orbit due to the outer orbit
    coeff_ein = 1.5 * epsilon_out * ein * one_minus_einsq_sqrt 
    coeff_hin = L_in * 1.5 * epsilon_out
    coeff_ein *= aoutoverrout3
    coeff_hin *= aoutoverrout3
    
    deinx_dt = norbit_in * coeff_ein * (5 * uindotrout * nincrossrout_x - nindotrout * uincrossrout_x - 2 * nincrossuin_x)
    
    deiny_dt = norbit_in * coeff_ein * (5 * uindotrout * nincrossrout_y - nindotrout * uincrossrout_y - 2 * nincrossuin_y)
    
    deinz_dt = norbit_in * coeff_ein * (5 * uindotrout * nincrossrout_z - nindotrout * uincrossrout_z - 2 * nincrossuin_z)
    
    dhinx_dt =  norbit_in * coeff_hin * (5 * ein_squared * uindotrout * uincrossrout_x - one_minus_einsq * nindotrout * nincrossrout_x)
    
    dhiny_dt =  norbit_in * coeff_hin * (5 * ein_squared * uindotrout * uincrossrout_y - one_minus_einsq * nindotrout * nincrossrout_y)
    
    dhinz_dt =  norbit_in * coeff_hin * (5 * ein_squared * uindotrout * uincrossrout_z - one_minus_einsq * nindotrout * nincrossrout_z)

    
    
    
    if (solve_for_spin_vector):
        if (triple_data['spin0']):
            dSpin0x_dt = 0
            
            dSpin0y_dt = 0
            
            dSpin0z_dt = 0
            
        if (triple_data['spin1']):
            dSpin1x_dt = 0
            
            dSpin1y_dt = 0
            
            dSpin1z_dt = 0
    else:
        if (triple_data['spin0']):
            dOmega0x_dt = 0
            
            dOmega0y_dt = 0
            
            dOmega0z_dt = 0

            dOmega0_dt = 0
            
        if (triple_data['spin1']):
            dOmega1x_dt = 0
            
            dOmega1y_dt = 0
            
            dOmega1z_dt = 0

            dOmega1_dt = 0
            
    if (octupole):
        epsilon_oct = (m0 - m1)/(m0 + m1) * (ain/aout) * aoutoverrout

        # For the inner orbit due to the outer orbit
        
        coeff_ein_oct = 0.9375 * epsilon_out * epsilon_oct * one_minus_einsq_sqrt
        
        deinx_dt += norbit_in * coeff_ein_oct * (16. * ein_squared * uindotrout * nincrossuin_x -  (1. - 8. * ein_squared) * nincrossrout_x
                                                 + 5. *  one_minus_einsq * nindotrout * nindotrout * nincrossrout_x
                                                 - 35. * ein_squared * uindotrout * uindotrout * nincrossrout_x
                                                 + 10. * ein_squared * nindotrout * uindotrout * uincrossrout_x)
        
        deiny_dt += norbit_in * coeff_ein_oct * (16. * ein_squared * uindotrout * nincrossuin_y -  (1. - 8. * ein_squared) * nincrossrout_y
                                                 + 5. *  one_minus_einsq * nindotrout * nindotrout * nincrossrout_y
                                                 - 35. * ein_squared * uindotrout * uindotrout * nincrossrout_y
                                                 + 10. * ein_squared * nindotrout * uindotrout * uincrossrout_y)
        
        deinz_dt += norbit_in * coeff_ein_oct * (16. * ein_squared * uindotrout * nincrossuin_z -  (1. - 8. * ein_squared) * nincrossrout_z
                                                 + 5. *  one_minus_einsq * nindotrout * nindotrout * nincrossrout_z
                                                 - 35. * ein_squared * uindotrout * uindotrout * nincrossrout_z
                                                 + 10. * ein_squared * nindotrout * uindotrout * uincrossrout_z)                                     


        coeff_hin_oct = L_in * 0.9375 * epsilon_out * epsilon_oct
        
        dhinx_dt += norbit_in * coeff_hin_oct * (10 *  ein * one_minus_einsq * nindotrout * uindotrout * nincrossrout_x
                                                 - (1. - 8. * ein_squared) * ein * uincrossrout_x
                                                 + 5. * ein * one_minus_einsq * nindotrout * nindotrout * uincrossrout_x
                                                 - 35. * ein_squared * ein * uindotrout * uindotrout * uincrossrout_x)
                                                 
        dhiny_dt += norbit_in * coeff_hin_oct * (10 *  ein * one_minus_einsq * nindotrout * uindotrout * nincrossrout_y
                                                 - (1. - 8. * ein_squared) * ein * uincrossrout_y
                                                 + 5. * ein * one_minus_einsq * nindotrout * nindotrout * uincrossrout_y
                                                 - 35. * ein_squared * ein * uindotrout * uindotrout * uincrossrout_y)

        dhinz_dt += norbit_in * coeff_hin_oct * (10 *  ein * one_minus_einsq * nindotrout * uindotrout * nincrossrout_z
                                                 - (1. - 8. * ein_squared) * ein * uincrossrout_z
                                                 + 5. * ein * one_minus_einsq * nindotrout * nindotrout * uincrossrout_z
                                                 - 35. * ein_squared * ein * uindotrout * uindotrout * uincrossrout_z)
        
    if (extra_forces_conservative) | (extra_forces_dissipative):

        deinx_dt += ein * ((Z0 + Z1 + ZGR) * vinx -  (Y0 + Y1) * ninx - (V0 + V1) * uinx)
        
        deiny_dt += ein * ((Z0 + Z1 + ZGR) * viny -  (Y0 + Y1) * niny - (V0 + V1) * uiny) 
        
        deinz_dt += ein * ((Z0 + Z1 + ZGR) * vinz -  (Y0 + Y1) * ninz - (V0 + V1) * uinz)
        
        dhinx_dt += hin * ((Y0 + Y1) * uinx - (X0 + X1) * vinx - (W0 + W1) * ninx)
        
        dhiny_dt += hin * ((Y0 + Y1) * uiny - (X0 + X1) * viny - (W0 + W1) * niny) 
        
        dhinz_dt += hin * ((Y0 + Y1) * uinz - (X0 + X1) * vinz - (W0 + W1) * ninz)
        

        if (triple_data['spin0']):
            
            if not (triple_data['spinorbit_align0']):
                dSpin0x_dt_tide = mu * hin * (-Y0 * uinx + X0 * vinx + W0 * ninx)
                
                dSpin0y_dt_tide = mu * hin * (-Y0 * uiny + X0 * viny + W0 * niny)
                
                dSpin0z_dt_tide = mu * hin * (-Y0 * uinz + X0 * vinz + W0 * ninz)


                if (solve_for_spin_vector):
                    dSpin0x_dt += dSpin0x_dt_tide
                    
                    dSpin0y_dt += dSpin0y_dt_tide
                    
                    dSpin0z_dt += dSpin0z_dt_tide
                else:
                    tau_omega = inf#0.5e6*365.25
                    Omegaref = 2 * pi/8.0

                    dOmega0x_dt += dSpin0x_dt_tide / I0 - dI0_dt * Omega0x / I0
                    
                    dOmega0y_dt += dSpin0y_dt_tide / I0 - dI0_dt * Omega0y / I0
                    
                    dOmega0z_dt += dSpin0z_dt_tide / I0 - dI0_dt * Omega0z / I0
                    
                    if (tau_omega != inf):
                        dOmega0x_dt += -(Omega0 - Omegaref)/tau_omega * Omega0x/Omega0
                        
                        dOmega0y_dt += -(Omega0 - Omegaref)/tau_omega * Omega0y/Omega0
                        
                        dOmega0z_dt += -(Omega0 - Omegaref)/tau_omega * Omega0z/Omega0
            else:
                dOmega0_dt =  mu * hin / I0 * W0
                
        if (triple_data['spin1']):
            if not (triple_data['spinorbit_align1']):
                dSpin1x_dt_tide = mu * hin * (-Y1 * uinx + X1 * vinx + W1 * ninx)
                
                dSpin1y_dt_tide = mu * hin * (-Y1 * uiny + X1 * viny + W1 * niny)
                
                dSpin1z_dt_tide = mu * hin * (-Y1 * uinz + X1 * vinz + W1 * ninz)

                if (solve_for_spin_vector):
                    dSpin1x_dt += dSpin1x_dt_tide
                    
                    dSpin1y_dt += dSpin1y_dt_tide
                    
                    dSpin1z_dt += dSpin1z_dt_tide
                else:
                    dOmega1x_dt += dSpin1x_dt_tide/I1 - dI1_dt * Omega1x / I1
                    
                    dOmega1y_dt += dSpin1y_dt_tide/I1 - dI1_dt * Omega1y / I1
                    
                    dOmega1z_dt += dSpin1z_dt_tide/I1 - dI1_dt * Omega1z / I1  

            else:
               dOmega1_dt =  mu * hin / I1 * W1

        
    ########################################################################
    # vector differential equations
    
    diffeq_list = []


    # for the inner binary
    if (triple_data['inner_orbit']):
        diffeq_list +=  [deinx_dt, 
                         deiny_dt, 
                         deinz_dt, 
                         dhinx_dt, 
                         dhiny_dt, 
                         dhinz_dt]


    
    # for the spin (specific) angular momenta
    if (triple_data['spin0']):
        if  (not triple_data['pseudosynch0']):
            if (solve_for_spin_vector):
                if not (triple_data['spinorbit_align0']):
                    diffeq_list += [dSpin0x_dt,
                                    dSpin0y_dt,
                                    dSpin0z_dt]
                else:
                    diffeq_list += [dSpin0_dt]
            else:
                if not (triple_data['spinorbit_align0']):
                    diffeq_list += [dOmega0x_dt,
                                    dOmega0y_dt,
                                    dOmega0z_dt]
                else:
                    diffeq_list += [dOmega0_dt]
                    
        
    if (triple_data['spin1']):
        if (not triple_data['pseudosynch1']):
            if (solve_for_spin_vector):
                if not (triple_data['spinorbit_align1']):
                    diffeq_list += [dSpin1x_dt,
                                    dSpin1y_dt,
                                    dSpin1z_dt]
                else:
                    diffeq_list += [dSpin1_dt]
            else:
                if not (triple_data['spinorbit_align1']):
                    diffeq_list += [dOmega1x_dt,
                                    dOmega1y_dt,
                                    dOmega1z_dt]
                else:
                    diffeq_list += [dOmega1_dt]



    #if the properties of the bodies are changing
    #if (dradius0_dt is not None) & (np.isfinite(dradius0_dt)):
    #    diffeq_list += [dR0_dt]
    #if (dradius1_dt is not None) & (np.isfinite(dradius1_dt)):
    #    diffeq_list += [dR1_dt]       
    

    return diffeq_list

