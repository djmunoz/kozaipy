import numpy as np
import scipy.integrate as integ
import triples
#import bsint

def threebody_ode_vf_tides(t,y,\
                           m0,m1,m2,
                           R0, R1,
                           rg_0, rg_1,
                           k2_0, k2_1,
                           tv0, tv1,
                           octupole,
                           extra_forces_conservative,
                           extra_forces_dissipative,
                           solve_for_spin_vector):


    # time-dependent variables
    #####################################################################
    #time-dependent variables ###########################################

    # get the active variables
    jj = 0
    # for the inner binary
    if (triples.triple_data['inner_orbit']):
        einx = y[jj+0]
        einy = y[jj+1]
        einz = y[jj+2]
        hinx = y[jj+3]
        hiny = y[jj+4]
        hinz = y[jj+5]
        jj+=6


    # for the outer orbit
    if (triples.triple_data['outer_orbit']):
        eoutx = y[jj+0]
        eouty = y[jj+1]
        eoutz = y[jj+2]
        houtx = y[jj+3]
        houty = y[jj+4]
        houtz = y[jj+5]
        jj+=6
    
    # for the spin (specific) angular momenta
    if (triples.triple_data['spin0']):
        if not  (triples.triple_data['spinorbit_align0']):
            Omega0x = y[jj+0]
            Omega0y = y[jj+1]
            Omega0z = y[jj+2]
            jj+=3
        else:
            Omega0 = y[jj+0]
            jj+=1
            
    if (triples.triple_data['spin1']):
        if not  (triples.triple_data['spinorbit_align1']):
            Omega1x = y[jj+0]
            Omega1y = y[jj+1]
            Omega1z = y[jj+2]
            jj+=3
        else:
            Omega1 = y[jj+0]
            jj+=1 
            

    ####################################################
    # Some quantities defined for convenience
    mu = m0 * m1 / (m0 + m1)
    
    ein_squared = einx**2 + einy**2 + einz**2
    ein = np.sqrt(ein_squared)
    one_minus_einsq = 1 - ein_squared
    one_minus_einsq_sqrt = np.sqrt(one_minus_einsq)
    one_minus_einsq_squared = one_minus_einsq * one_minus_einsq
    one_minus_einsq_fifth =  one_minus_einsq_squared * one_minus_einsq_squared * one_minus_einsq
    hin = np.sqrt(hinx**2 + hiny**2 + hinz**2)

    eout_squared = eoutx**2 + eouty**2 + eoutz**2
    hout = np.sqrt(houtx**2 + houty**2 + houtz**2)
    eout = np.sqrt(eout_squared)
    one_minus_eoutsq = 1 - eout_squared
    one_minus_eoutsq_sqrt = np.sqrt(one_minus_eoutsq)
    one_minus_eoutsq_squared = one_minus_eoutsq * one_minus_eoutsq

    Gm_in =  triples.constants.G * (m0 + m1)
    Gm_out =  triples.constants.G * (m0 + m1 + m2)
    ain = hin * hin / (1 - ein * ein) / Gm_in
    aout = hout * hout / (1 - eout * eout) / Gm_out
    norbit_in = np.sqrt(Gm_in/ain/ain/ain)
    norbit_out = np.sqrt(Gm_out/aout/aout/aout)
    L_in = np.sqrt(Gm_in * ain)
    L_out = np.sqrt(Gm_out * aout)
    
    # unit vectors
    uinx, uiny, uinz = einx/ein, einy/ein, einz/ein
    ninx, niny, ninz = hinx/hin, hiny/hin, hinz/hin   
    vinx = (niny * uinz - ninz * uiny)
    viny = (ninz * uinx - ninx * uinz)
    vinz = (ninx * uiny - niny * uinx)
    noutx, nouty, noutz = houtx/hout, houty/hout, houtz/hout
    uoutx, uouty, uoutz = eoutx/eout, eouty/eout, eoutz/eout  
    nindotnout =  noutx * ninx + nouty * niny + noutz * ninz
    uindotnout =  noutx * uinx + nouty * uiny + noutz * uinz
    uindotuout =  uoutx * uinx + uouty * uiny + uoutz * uinz
    nindotuout =  uoutx * ninx + uouty * niny + uoutz * ninz
    nincrossuin_x = niny * uinz - ninz * uiny
    nincrossuin_y = ninz * uinx - ninx * uinz
    nincrossuin_z = ninx * uiny - niny * uinx
    nincrossuout_x = niny * uoutz - ninz * uouty
    nincrossuout_y = ninz * uoutx - ninx * uoutz
    nincrossuout_z = ninx * uouty - niny * uoutx   
    uincrossuout_x = uiny * uoutz - uinz * uouty
    uincrossuout_y = uinz * uoutx - uinx * uoutz
    uincrossuout_z = uinx * uouty - uiny * uoutx   
    uoutcrossnout_x = uouty * noutz - uoutz * nouty
    uoutcrossnout_y = uoutz * noutx - uoutx * noutz
    uoutcrossnout_z = uoutx * nouty - uouty * noutx
    nincrossnout_x = niny * noutz - ninz * nouty
    nincrossnout_y = ninz * noutx - ninx * noutz
    nincrossnout_z = ninx * nouty - niny * noutx
    uincrossnout_x = uiny * noutz - uinz * nouty
    uincrossnout_y = uinz * noutx - uinx * noutz
    uincrossnout_z = uinx * nouty - uiny * noutx

    ein_fourth = ein_squared * ein_squared
    ein_sixth = ein_fourth * ein_squared
    f2 = 1 + 7.5 * ein_squared + 5.625 * ein_fourth + 0.3125 * ein_sixth
    f3 = 1 + 3.75 * ein_squared + 1.875 * ein_fourth + 0.078125 * ein_sixth
    f4 = 1 + 1.5 * ein_squared + 0.125 * ein_fourth
    f5 = 1 + 3.0 * ein_squared + 0.375 * ein_fourth

    if (ein < 1e-10):
        ein = 0
        ein_squared = 0
        one_minus_einsq = 1
        one_minus_einsq_sqrt = 1
        one_minus_einsq_squared = 1
        one_minus_einsq_fifth =  1
        ein_fourth = 0
        ein_sixth = 0
        f2, f3, f4, f5 = 1, 1, 1, 1
        
    if (triples.triple_data['spin0']):
        I0 = rg_0 * m0 * R0 * R0
        if not  (triples.triple_data['spinorbit_align0']):
            Omega0_u = (Omega0x * uinx + Omega0y * uiny + Omega0z * uinz)
            Omega0_n = (Omega0x * ninx + Omega0y * niny + Omega0z * ninz)
            Omega0_v = (Omega0x * vinx + Omega0y * viny + Omega0z * vinz)            
            Omega0 = np.sqrt(Omega0_u**2 + Omega0_v**2 + Omega0_n**2)
        else:
            Omega0_u = 0
            Omega0_v = 0
            Omega0_n = Omega0
            
    elif (triples.triple_data['pseudosynch0']):
        Omega0_u = 0
        Omega0_n = f2/f5/one_minus_einsq/one_minus_einsq_sqrt * norbit_in
        Omega0_v = 0
        Omega0 = Omega0_n
    else:
        Omega0_u, Omega0_v, Omega0_n, Omega0 = 0, 0, 0, 0
        
    if (triples.triple_data['spin1']):
        I1 = rg_1 * m1 * R1 * R1
        if not  (triples.triple_data['spinorbit_align1']):
            Omega1_u = (Omega1x * uinx + Omega1y * uiny + Omega1z * uinz)
            Omega1_n = (Omega1x * ninx + Omega1y * niny + Omega1z * ninz)
            Omega1_v = (Omega1x * vinx + Omega1y * viny + Omega1z * vinz)
            Omega1 = np.sqrt(Omega1_u**2 + Omega1_v**2 + Omega1_n**2)
        else:
            Omega1_u = 0
            Omega1_v = 0
            Omega1_n = Omega1
    elif (triples.triple_data['pseudosynch1']):
        Omega1_u = 0
        Omega1_n = f2/f5/one_minus_einsq/one_minus_einsq_sqrt * norbit_in
        Omega1_v = 0
        Omega1 = Omega1_n
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
                                                                  + 15 * triples.constants.G * m1 / ain**3 * f4 / one_minus_einsq_fifth)

        V1 = 0
                      
        W1 = 0
        
        X1 = -1.0/norbit_in * m0 * k2_1 * size_ratio1_fifth / mu  *  Omega1_n *  Omega1_u / one_minus_einsq_squared
                
        Y1 = -1.0/norbit_in * m0 * k2_1 * size_ratio1_fifth / mu  *  Omega1_n *  Omega1_v / one_minus_einsq_squared
        
        Z1 = 1.0/norbit_in * m0 * k2_1 * size_ratio1_fifth /mu * (0.5*(2 * Omega1_n**2 - Omega1_u**2 - Omega1_v**2) / one_minus_einsq_squared \
                                                                  + 15 * triples.constants.G * m0 / ain**3 * f4 / one_minus_einsq_fifth)
        
        
        ZGR = 3 * triples.constants.G * (m0 + m1) * norbit_in / ain / triples.constants.CLIGHT / triples.constants.CLIGHT / one_minus_einsq
    

        

        if (extra_forces_dissipative):

            tf0 = tv0/9 / size_ratio0_eighth * m0**2 / ((m0 + m1)*m1) / (1 + 2 * k2_0)**2 
            tf1 = tv1/9 / size_ratio1_eighth * m1**2 / ((m0 + m1)*m0) / (1 + 2 * k2_1)**2 
            
            timelag0 = 1.5 / tv0 * R0 * R0 * R0 / triples.constants.G / m0 * (1 + 2 * k2_0)**2/ k2_0
            timelag1 = 1.5 / tv1 * R1 * R1 * R1 / triples.constants.G / m1 * (1 + 2 * k2_1)**2/ k2_1

            #print timelag0/365.25,timelag1/365.25
            
            #tf0 = m0 /m1 / size_ratio0_fifth / norbit_in /norbit_in / timelag0 /6 / k2_0
            #tf1 = m1 /m0 / size_ratio1_fifth / norbit_in /norbit_in / timelag1 /6 / k2_1


            
            #if (t > 3.01e9 *365.25): print t,tf0,tf1,size_ratio0,size_ratio1 
            
            Q0 = 4.0/3 * k2_0 / (1 + 2 * k2_0)**2 * triples.constants.G * m0/ R0**3 * tv0 / norbit_in
            Q1 = 4.0/3 * k2_1 / (1 + 2 * k2_1)**2 * triples.constants.G * m1/ R1**3 * tv1 / norbit_in
            
            V0 += 9.0 / tf0 * (f3 / one_minus_einsq_fifth/one_minus_einsq/one_minus_einsq_sqrt - \
                               11.0/18 * Omega0_n / norbit_in * f4 / one_minus_einsq_fifth)

            W0 += 1.0 / tf0 * (f2 / one_minus_einsq_fifth/one_minus_einsq/one_minus_einsq_sqrt - \
                               Omega0_n / norbit_in * f5 / one_minus_einsq_fifth)    
            
            X0 += -1.0/norbit_in * Omega0_v /2/tf0 * (1 + 4.5 * ein_squared + 0.625 * ein_fourth) / one_minus_einsq_fifth
        
            Y0 += 1.0/norbit_in * Omega0_u /2/tf0 * (1 + 1.5 * ein_squared + 0.125 * ein_fourth) / one_minus_einsq_fifth
            

        
            V1 += 9.0 / tf1 * (f3 / one_minus_einsq_fifth/one_minus_einsq/one_minus_einsq_sqrt - \
                               11.0/18 * Omega1_n /norbit_in * f4 / one_minus_einsq_fifth)
            
            W1 += 1.0 / tf1 * (f2 / one_minus_einsq_fifth/one_minus_einsq/one_minus_einsq_sqrt - \
                               Omega1_n / norbit_in * f5 /one_minus_einsq_fifth)    
            
            X1 += -1.0/norbit_in * Omega1_v /2/tf1 * (1 + 4.5 * ein_squared + 0.625 * ein_fourth) / one_minus_einsq_fifth
            
            Y1 += 1.0/norbit_in * Omega1_u /2/tf1 * (1 + 1.5 * ein_squared + 0.125 * ein_fourth) / one_minus_einsq_fifth

    else:
        V0, W0, X0, Y0, Z0 = 0, 0, 0, 0, 0
        V1, W1, X1, Y1, Z1 = 0, 0, 0, 0, 0
        ZGR = 0

    ##########################################################################
    # System of differential equations


    # Equations of motion at the quadrupole level:
    
    epsilon_in = 0.5 * (m0 * m1/(m0 + m1)/(m0 + m1)) * ain * ain / aout / aout / one_minus_eoutsq_sqrt / one_minus_eoutsq
    epsilon_out = m2 / (m0 + m1) * ain * ain * ain / aout / aout / aout / one_minus_eoutsq / one_minus_eoutsq_sqrt 


    # For the inner orbit

    coeff_ein = 0.75 * epsilon_out * ein * one_minus_einsq_sqrt 
    
    deinx_dt =  norbit_in * coeff_ein * (nindotnout * uincrossnout_x  + 2 * nincrossuin_x - 5 * uindotnout * nincrossnout_x)

    deiny_dt =  norbit_in * coeff_ein * (nindotnout * uincrossnout_y  + 2 * nincrossuin_y - 5 * uindotnout * nincrossnout_y)

    deinz_dt =  norbit_in * coeff_ein * (nindotnout * uincrossnout_z  + 2 * nincrossuin_z - 5 * uindotnout * nincrossnout_z)

    
    coeff_hin = L_in * 0.75 * epsilon_out
    
    dhinx_dt =  norbit_in * coeff_hin * (one_minus_einsq * nindotnout * nincrossnout_x - 5 * ein_squared * uindotnout * uincrossnout_x)
                                      
    dhiny_dt =  norbit_in * coeff_hin * (one_minus_einsq * nindotnout * nincrossnout_y - 5 * ein_squared * uindotnout * uincrossnout_y)
                                      
    dhinz_dt =  norbit_in * coeff_hin * (one_minus_einsq * nindotnout * nincrossnout_z - 5 * ein_squared * uindotnout * uincrossnout_z)
    

    # For the outer orbit

    coeff_eout = 1.5 * epsilon_in * eout / one_minus_eoutsq_sqrt 
    
    deoutx_dt = norbit_out * coeff_eout * (-one_minus_einsq * nindotnout * nincrossuout_x\
                                           + 5 * ein_squared * uindotnout * uincrossuout_x \
                                           + 0.5 * ((1 - 6 * ein_squared) + \
                                                    25 * ein_squared * uindotnout * uindotnout \
                                                    - 5 * one_minus_einsq * nindotnout * nindotnout) * uoutcrossnout_x)
    
    deouty_dt = norbit_out * coeff_eout * (-one_minus_einsq * nindotnout * nincrossuout_y\
                                           + 5 * ein_squared * uindotnout * uincrossuout_y \
                                           + 0.5 * ((1 - 6 * ein_squared) + \
                                                    25 * ein_squared * uindotnout * uindotnout \
                                                    - 5 * one_minus_einsq * nindotnout * nindotnout) * uoutcrossnout_y)
    
    deoutz_dt = norbit_out * coeff_eout * (-one_minus_einsq * nindotnout * nincrossuout_z\
                                           + 5 * ein_squared * uindotnout * uincrossuout_z \
                                           + 0.5 * ((1 - 6 * ein_squared) + \
                                                    25 * ein_squared * uindotnout * uindotnout \
                                                    - 5 * one_minus_einsq * nindotnout * nindotnout) * uoutcrossnout_z)
    
    coeff_hout = L_out * 1.5 * epsilon_in 
    
    dhoutx_dt = norbit_out * coeff_hout * (-one_minus_einsq * nindotnout * nincrossnout_x + 5 * ein_squared * uindotnout * uincrossnout_x)
    
    dhouty_dt = norbit_out * coeff_hout * (-one_minus_einsq * nindotnout * nincrossnout_y + 5 * ein_squared * uindotnout * uincrossnout_y)
    
    dhoutz_dt = norbit_out * coeff_hout * (-one_minus_einsq * nindotnout * nincrossnout_z + 5 * ein_squared * uindotnout * uincrossnout_z) 
    
    if (solve_for_spin_vector):
        if (triples.triple_data['spin0']):
            dSpin0x_dt = 0
            
            dSpin0y_dt = 0
            
            dSpin0z_dt = 0
            
        if (triples.triple_data['spin1']):
            dSpin1x_dt = 0
            
            dSpin1y_dt = 0
            
            dSpin1z_dt = 0
    else:
        if (triples.triple_data['spin0']):
            dOmega0x_dt = 0
            
            dOmega0y_dt = 0
            
            dOmega0z_dt = 0
            
        if (triples.triple_data['spin1']):
            dOmega1x_dt = 0
            
            dOmega1y_dt = 0
            
            dOmega1z_dt = 0

            
            
    if (octupole):
        epsilon_oct = (m0 - m1)/(m0 + m1) * (ain/aout) 

        # For the inner orbit
        
        coeff_ein_oct = -1.171875 * epsilon_out * epsilon_oct * eout / one_minus_eoutsq * one_minus_einsq_sqrt
        
        deinx_dt += norbit_in * coeff_ein_oct * ((2 * ein_squared * uindotnout * nindotnout) * uincrossuout_x +\
                                                 0.2 * (8 * ein_squared - 1 - \
                                                        35 * ein_squared * uindotnout * uindotnout + \
                                                        5 * one_minus_einsq * nindotnout * nindotnout) * nincrossuout_x + \
                                                 2 * ein_squared * (uindotuout * nindotnout + \
                                                                    uindotnout * nindotuout) * uincrossnout_x + \
                                                 2 * (one_minus_einsq * nindotnout * nindotuout - \
                                                      7 *  ein_squared * uindotnout * uindotuout) * nincrossnout_x +\
                                                 3.2 * ein_squared  * uindotuout * nincrossuin_x)
        
        deiny_dt += norbit_in * coeff_ein_oct * ((2 * ein_squared * uindotnout * nindotnout) * uincrossuout_y +\
                                                  0.2 * (8 * ein_squared - 1 - \
                                                         35 * ein_squared * uindotnout * uindotnout + \
                                                         5 * one_minus_einsq * nindotnout * nindotnout) * nincrossuout_y + \
                                                 2 * ein_squared * (uindotuout * nindotnout + \
                                                                    uindotnout * nindotuout) * uincrossnout_y + \
                                                 2 * (one_minus_einsq * nindotnout * nindotuout - \
                                                       7 *  ein_squared * uindotnout * uindotuout) * nincrossnout_y +\
                                                 3.2 * ein_squared  * uindotuout * nincrossuin_y)
        
        deinz_dt += norbit_in * coeff_ein_oct * ((2 * ein_squared * uindotnout * nindotnout) * uincrossuout_z +\
                                                  0.2 * (8 * ein_squared - 1 - \
                                                         35 * ein_squared * uindotnout * uindotnout + \
                                                         5 * one_minus_einsq * nindotnout * nindotnout) * nincrossuout_z + \
                                                 2 * ein_squared * (uindotuout * nindotnout + \
                                                                    uindotnout * nindotuout) * uincrossnout_z + \
                                                 2 * (one_minus_einsq * nindotnout * nindotuout - \
                                                       7 *  ein_squared * uindotnout * uindotuout) * nincrossnout_z +\
                                                 3.2 * ein_squared  * uindotuout * nincrossuin_z)


        coeff_hin_oct = -L_in * 1.171875 * epsilon_out * epsilon_oct * eout / one_minus_eoutsq * ein
        
        dhinx_dt += norbit_in * coeff_hin_oct * (2 * one_minus_einsq * (uindotuout * nindotnout + \
                                                                        uindotnout * nindotuout) * nincrossnout_x +\
                                                 2 * (one_minus_einsq * nindotuout * nindotnout - \
                                                      7 * ein_squared * uindotuout * uindotnout) * uincrossnout_x +\
                                                 2 * one_minus_einsq * uindotnout * nindotnout * nincrossuout_x +\
                                                 0.2 * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout +\
                                                        5 * one_minus_einsq * nindotnout * nindotnout) * uincrossuout_x)
                    
        dhiny_dt += norbit_in * coeff_hin_oct * (2 * one_minus_einsq * (uindotuout * nindotnout + \
                                                                        uindotnout * nindotuout) * nincrossnout_y +\
                                                 2 * (one_minus_einsq * nindotuout * nindotnout - \
                                                      7 * ein_squared * uindotuout * uindotnout) * uincrossnout_y +\
                                                 2 * one_minus_einsq * uindotnout * nindotnout * nincrossuout_y +\
                                                 0.2 * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout +\
                                                        5 * one_minus_einsq * nindotnout * nindotnout) * uincrossuout_y)
        
        dhinz_dt += norbit_in * coeff_hin_oct * (2 * one_minus_einsq * (uindotuout * nindotnout + \
                                                                        uindotnout * nindotuout) * nincrossnout_z +\
                                                 2 * (one_minus_einsq * nindotuout * nindotnout - \
                                                      7 * ein_squared * uindotuout * uindotnout) * uincrossnout_z +\
                                                 2 * one_minus_einsq * uindotnout * nindotnout * nincrossuout_z +\
                                                 0.2 * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout +\
                                                        5 * one_minus_einsq * nindotnout * nindotnout) * uincrossuout_z)

        # For the outer orbit

        coeff_eout_oct = -2.34375 * epsilon_in * epsilon_oct * eout / one_minus_eoutsq / one_minus_eoutsq_sqrt * ein
        
        deoutx_dt += norbit_out * coeff_eout_oct * (-2 * eout * one_minus_einsq * (uindotnout * nindotuout +\
                                                                                   nindotnout * uindotuout) * nincrossuout_x\
                                                    -2 * one_minus_eoutsq/eout * one_minus_einsq * uindotnout * nindotnout * nincrossnout_x\
                                                    -2 * eout * (one_minus_einsq * nindotuout * nindotnout -\
                                                                 7 * ein_squared * uindotuout * uindotnout) * uincrossuout_x \
                                                    -one_minus_eoutsq/eout * 0.2 *(8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout\
                                                                                   + 5 * one_minus_einsq * nindotnout * nindotnout) * uincrossnout_x\
                                                    - eout * (0.4 *(1 - 8 * ein_squared) * uindotuout +\
                                                              14 * one_minus_einsq * uindotnout * nindotuout * nindotnout +\
                                                              1.4 * uindotuout * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout+\
                                                                                  5 * one_minus_einsq * nindotnout * nindotnout)) * uoutcrossnout_x)
        
        deouty_dt += norbit_out * coeff_eout_oct * (-2 * eout * one_minus_einsq * (uindotnout * nindotuout +\
                                                                                    nindotnout * uindotuout) * nincrossuout_y\
                                                    -2 * one_minus_eoutsq/eout * one_minus_einsq * uindotnout * nindotnout * nincrossnout_y\
                                                    -2 * eout * (one_minus_einsq * nindotuout * nindotnout -\
                                                                 7 * ein_squared * uindotuout * uindotnout) * uincrossuout_y \
                                                    -one_minus_eoutsq/eout * 0.2 *(8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout\
                                                                                   + 5 * one_minus_einsq * nindotnout * nindotnout) * uincrossnout_y\
                                                    - eout * (0.4 *(1 - 8 * ein_squared) * uindotuout +\
                                                              14 * one_minus_einsq * uindotnout * nindotuout * nindotnout +\
                                                              1.4 * uindotuout * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout+\
                                                                                  5 * one_minus_einsq * nindotnout * nindotnout)) * uoutcrossnout_y)
        
        deoutz_dt += norbit_out * coeff_eout_oct * (-2 * eout * one_minus_einsq * (uindotnout * nindotuout +\
                                                                                    nindotnout * uindotuout) * nincrossuout_z\
                                                    -2 * one_minus_eoutsq/eout * one_minus_einsq * uindotnout * nindotnout * nincrossnout_z\
                                                    -2 * eout * (one_minus_einsq * nindotuout * nindotnout -\
                                                                 7 * ein_squared * uindotuout * uindotnout) * uincrossuout_z \
                                                    -one_minus_eoutsq/eout * 0.2 *(8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout\
                                                                                   + 5 * one_minus_einsq * nindotnout * nindotnout) * uincrossnout_z\
                                                    - eout * (0.4 *(1 - 8 * ein_squared) * uindotuout +\
                                                              14 * one_minus_einsq * uindotnout * nindotuout * nindotnout +\
                                                              1.4 * uindotuout * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout+\
                                                                                  5 * one_minus_einsq * nindotnout * nindotnout)) * uoutcrossnout_z)                                                                                          

        coeff_hout_oct = -L_out * 2.34375 * epsilon_in * epsilon_oct * eout / one_minus_eoutsq  * ein
        
        dhoutx_dt += norbit_out * coeff_hout_oct * (-2 * one_minus_einsq * (uindotnout * nindotuout + uindotuout * nindotnout) * nincrossnout_x\
                                                    -2 * one_minus_einsq * uindotnout * nindotnout * nincrossuout_x\
                                                    -2 * (one_minus_einsq * nindotuout * nindotnout \
                                                          - 7 * ein_squared * uindotuout * uindotnout) * uincrossnout_x\
                                                    -0.2 * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout \
                                                            + 5 * one_minus_einsq * nindotnout * nindotnout) * uincrossuout_x)
          
        dhouty_dt += norbit_out * coeff_hout_oct * (-2 * one_minus_einsq * (uindotnout * nindotuout + uindotuout * nindotnout) * nincrossnout_y\
                                                    -2 * one_minus_einsq * uindotnout * nindotnout * nincrossuout_y\
                                                    -2 * (one_minus_einsq * nindotuout * nindotnout \
                                                          - 7 * ein_squared * uindotuout * uindotnout) * uincrossnout_y\
                                                    -0.2 * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout \
                                                            + 5 * one_minus_einsq * nindotnout * nindotnout) * uincrossuout_y)        
        
        dhoutz_dt += norbit_out * coeff_hout_oct * (-2 * one_minus_einsq * (uindotnout * nindotuout + uindotuout * nindotnout) * nincrossnout_z\
                                                    -2 * one_minus_einsq * uindotnout * nindotnout * nincrossuout_z\
                                                    -2 * (one_minus_einsq * nindotuout * nindotnout \
                                                          - 7 * ein_squared * uindotuout * uindotnout) * uincrossnout_z\
                                                    -0.2 * (8 * ein_squared - 1 - 35 * ein_squared * uindotnout * uindotnout \
                                                            + 5 * one_minus_einsq * nindotnout * nindotnout) * uincrossuout_z)
          


    
    if (extra_forces_conservative):

        deinx_dt += ein * ((Z0 + Z1 + ZGR) * vinx -  (Y0 + Y1) * ninx - (V0 + V1) * uinx)
        
        deiny_dt += ein * ((Z0 + Z1 + ZGR) * viny -  (Y0 + Y1) * niny - (V0 + V1) * uiny) 
        
        deinz_dt += ein * ((Z0 + Z1 + ZGR) * vinz -  (Y0 + Y1) * ninz - (V0 + V1) * uinz)
        
        dhinx_dt += hin * ((Y0 + Y1) * uinx - (X0 + X1) * vinx - (W0 + W1) * ninx)
        
        dhiny_dt += hin * ((Y0 + Y1) * uiny - (X0 + X1) * viny - (W0 + W1) * niny) 
        
        dhinz_dt += hin * ((Y0 + Y1) * uinz - (X0 + X1) * vinz - (W0 + W1) * ninz)
        

        if (triples.triple_data['spin0']):
            if not (triples.triple_data['spinorbit_align0']):
                if (solve_for_spin_vector):
                    dSpin0x_dt += mu * hin * (-Y0 * uinx + X0 * vinx + W0 * ninx)
                    
                    dSpin0y_dt += mu * hin * (-Y0 * uiny + X0 * viny + W0 * niny)
                    
                    dSpin0z_dt += mu * hin * (-Y0 * uinz + X0 * vinz + W0 * ninz)
                else:
                    dOmega0x_dt += mu * hin / I0 * (-Y0 * uinx + X0 * vinx + W0 * ninx) 
                    
                    dOmega0y_dt += mu * hin / I0 * (-Y0 * uiny + X0 * viny + W0 * niny) 
                    
                    dOmega0z_dt += mu * hin / I0 * (-Y0 * uinz + X0 * vinz + W0 * ninz) 
            else:
                dOmega0_dt =  mu * hin / I0 * W0
                    
        if (triples.triple_data['spin1']):
            if not (triples.triple_data['spinorbit_align1']):
                if (solve_for_spin_vector):
                    dSpin1x_dt += mu * hin * (-Y1 * uinx + X1 * vinx + W1 * ninx)
                    
                    dSpin1y_dt += mu * hin * (-Y1 * uiny + X1 * viny + W1 * niny)
                    
                    dSpin1z_dt += mu * hin * (-Y1 * uinz + X1 * vinz + W1 * ninz)
                else:
                    dOmega1x_dt += mu * hin / I1 * (-Y1 * uinx + X1 * vinx + W1 * ninx) 
                    
                    dOmega1y_dt += mu * hin / I1 * (-Y1 * uiny + X1 * viny + W1 * niny) 
                    
                    dOmega1z_dt += mu * hin / I1 * (-Y1 * uinz + X1 * vinz + W1 * ninz) 
            else:
               dOmega1_dt =  mu * hin / I1 * W1
               
    ########################################################################
    # vector differential equations
    
    diffeq_list = []


    # for the inner binary
    if (triples.triple_data['inner_orbit']):
        diffeq_list +=  [deinx_dt, 
                         deiny_dt, 
                         deinz_dt, 
                         dhinx_dt, 
                         dhiny_dt, 
                         dhinz_dt]


    # for the outer orbit
    if (triples.triple_data['outer_orbit']):
        diffeq_list += [deoutx_dt, 
                        deouty_dt, 
                        deoutz_dt, 
                        dhoutx_dt, 
                        dhouty_dt, 
                        dhoutz_dt]
    
    # for the spin (specific) angular momenta
    if (triples.triple_data['spin0']):
        if (not triples.triple_data['pseudosynch0']):
            if not (triples.triple_data['spinorbit_align0']):
                diffeq_list += [dOmega0x_dt,
                                dOmega0y_dt,
                                dOmega0z_dt]
            else:
                diffeq_list += [dOmega0_dt]
            
            
        
    if (triples.triple_data['spin1']):
        if not (triples.triple_data['pseudosynch1']):
            if  not (triples.triple_data['spinorbit_align1']):
                diffeq_list += [dOmega1x_dt,
                                dOmega1y_dt,
                                dOmega1z_dt]
            else:
                diffeq_list += [dOmega1_dt]
            

    return diffeq_list



def threebody_ode_vf_tides_modified(y,t,\
                                    m0,m1,m2,
                                    R0, R1,
                                    rg_0, rg_1,
                                    k2_0, k2_1,
                                    tv0, tv1,
                                    octupole,
                                    extra_forces_conservative,
                                    extra_forces_dissipative,
                                    solve_for_spin_vector):

    
    return threebody_ode_vf_tides(t,y,\
                                  m0,m1,m2,
                                  R0, R1,
                                  rg_0, rg_1,
                                  k2_0, k2_1,
                                  tv0, tv1,
                                  octupole,
                                  extra_forces_conservative,
                                  extra_forces_dissipative,
                                  solve_for_spin_vector)

