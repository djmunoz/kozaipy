import numpy as np
import scipy.integrate as integ
import triples

def integrate_triple_system(ics,timemax,Nevals,
                            m0,m1,m2,
                            R0 = None,R1 = None,
                            I0 = None, I1 = None,
                            k20 = None, k21  = None,
                            tv0 = None, tv1  = None,
                            octupole_potential = True,
                            short_range_forces_conservative= False,
                            short_range_forces_dissipative = False,
                            tol = 1.0e-13):

    atol = tol   # this set is probably fine
    rtol = tol/10.0  # this is probably fine
    #atol = 1.0e-10
    #rtol = 1.0e-11
    #atol = 1.0e-11
    #rtol = 1.0e-12

    t = np.linspace(0,timemax,Nevals)
    sol=integ.odeint(threebody_ode_vf,ics,t,
                     args=(m0,m1,m2,R0,R1,I0,I1,k20,k21,tv0,tv1,
                           octupole_potential,\
                           short_range_forces_conservative,short_range_forces_dissipative,atol,),
                     atol=atol,rtol=rtol,mxstep=100000000,hmin=0.0000001)#,mxords=15)

    
    return np.column_stack((t,sol))
    
    
def threebody_ode_vf(y,t,m0,m1,m2,
                     R0 = None,R1 = None,
                     I0 = None, I1 = None,
                     k20 = None, k21  = None,
                     tv0 = None, tv1  = None,
                     octupole = True,
                     extra_forces_conservative = False,
                     extra_forces_dissipative = False,
                     tol=1.0e-12):

    # 20 time-dependent variables
    #####################################################################
    #time-dependent variables ###########################################

    # for the inner binary
    einx = y[0]
    einy = y[1]
    einz = y[2]
    hinx = y[3]
    hiny = y[4]
    hinz = y[5]



    # for the outer orbit
    eoutx = y[6]
    eouty = y[7]
    eoutz = y[8]
    houtx = y[9]
    houty = y[10]
    houtz = y[11]

    
    # for the spin angular momenta
    S0x = y[12]
    S0y = y[13]
    S0z = y[14]
    S1x = y[15]
    S1y = y[16]
    S1z = y[17]
        

    ####################################################
    # Some quantities defined for convenience
    
    ein_squared = einx**2 + einy**2 + einz**2
    ein = np.sqrt(ein_squared)
    one_minus_einsq = 1 - ein_squared
    one_minus_einsq_sqrt = np.sqrt(one_minus_einsq)
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

    if (ein < 1e-7):
        ein = 0
        ein_squared = 0

    Omega0_u = (S0x * uinx + S0y * uiny + S0z * uinz) / I0
    Omega0_n = (S0x * ninx + S0y * niny + S0z * ninz) / I0
    Omega0_v = (S0x * vinx + S0y * viny + S0z * vinz) / I0
    
    Omega1_u = (S1x * uinx + S1y * uiny + S1z * uinz) / I1
    Omega1_n = (S1x * ninx + S1y * niny + S1z * ninz) / I1
    Omega1_v = (S1x * vinx + S1y * viny + S1z * vinz) / I1

    
    if (extra_forces_conservative):
        
        Qstar = 0.028
        Qplanet = 0.51
        tvstar =   2.0e4 # in days, about 55 years
        tvplanet = 0.365242  # in days, about 0.001 years
        
        Q0, Q1 = Qstar, Qplanet
        
        tv0, tv1 = tvstar, tvplanet
        
        k0 = 0.5 * Q0/(1 - Q0)
        k1 = 0.5 * Q1/(1 - Q1)
        
        ein_fourth = ein_squared * ein_squared
        ein_sixth = ein_fourth * ein_squared


        V0 = 0
                      
        W0 = 0
              
        X0 = -1.0/norbit_in * m1 * k0 * (R0/ain)**5 / mu  *  Omega0_n *  Omega0_u / one_minus_einsq**2 
        
        Y0 = -1.0/norbit_in * m1 * k0 * (R0/ain)**5 / mu  *  Omega0_n *  Omega0_v / one_minus_einsq**2
        
        Z0 = 1.0/norbit_in * m1 * k0 * (R0/ain)**5 /mu * (0.5 * (2 * Omega0_n**2 - Omega0_u**2 - Omega0_v**2) / one_minus_einsq**2 \
                                                        +15 * triples.constants.G * m1 / ain**3 *(1 + 1.5*ein_squared + 0.125*ein_fourth) / one_minus_einsq**5)

        V1 = 0
                      
        W1 = 0
        
        X1 = -1.0/norbit_in * m0 * k1 * (R1/ain)**5 / mu  *  Omega1_n *  Omega1_u / one_minus_einsq**2
                
        Y1 = -1.0/norbit_in * m0 * k1 * (R1/ain)**5 / mu  *  Omega1_n *  Omega1_v / one_minus_einsq**2
        
        Z1 = 1.0/norbit_in * m0 * k1 * (R1/ain)**5 /mu * (0.5*(2 * Omega1_n**2 - Omega1_u**2 - Omega1_v**2) / one_minus_einsq**2 \
                                                        +15 * triples.constants.G * m0 / ain**3 *(1 + 1.5*ein_squared + 0.125*ein_fourth) / one_minus_einsq**5)
        
        
        ZGR = 3 * triples.constants.G * (m0 + m1) * norbit_in / ain / triples.constants.CLIGHT / triples.constants.CLIGHT / one_minus_einsq
    

        

        if (extra_forces_dissipative):

            tf0 = tv0/9 * (ain/R0)**8 * m0**2 / ((m0 + m1)*m1) / (1 + 2 * k0)**2 
            
            tf1 = tv1/9 * (ain/R1)**8 * m1**2 / ((m0 + m1)*m0) / (1 + 2 * k1)**2 
        


            
            V0 += 9.0 / tf0 * ((1 + 15.0/4 * ein_squared + 15.0/8 * ein_fourth + 5.0/64 * ein_sixth)/one_minus_einsq**6.5 - \
                               11.0/18 * Omega0_n / norbit_in * (1 + 1.5 * ein_squared + 0.125 * ein_fourth)/one_minus_einsq**5)
        
            W0 += 1.0 / tf0 * ((1 + 15.0/2 * ein_squared + 45.0/8 * ein_fourth + 5.0/16 * ein_sixth)/one_minus_einsq**6.5 - \
                              Omega0_n / norbit_in * (1 + 3 * ein_squared + 0.375 * ein_fourth)/one_minus_einsq**5)    
            
            X0 += -1.0/norbit_in * Omega0_v /2/tf0 * (1 + 4.5 * ein_squared + 0.625 * ein_fourth) / one_minus_einsq**5
        
            Y0 += 1.0/norbit_in * Omega0_u /2/tf0 * (1 + 1.5 * ein_squared + 0.125 * ein_fourth) / one_minus_einsq**5
            

        
            V1 += 9.0 / tf1 * ((1 + 15.0/4 * ein_squared + 15.0/8 * ein_fourth + 5.0/64 * ein_sixth)/one_minus_einsq**6.5 - \
                              11.0/18 * Omega1_n /norbit_in * (1 + 1.5 * ein_squared + 0.125 * ein_fourth)/one_minus_einsq**5)
            
            W1 += 1.0 / tf1 * ((1 + 15.0/2 * ein_squared + 45.0/8 * ein_fourth + 5.0/16 * ein_sixth)/one_minus_einsq**6.5 - \
                              Omega1_n / norbit_in * (1 + 3 * ein_squared + 0.375 * ein_fourth)/one_minus_einsq**5)    
        
            X1 += 1.0/norbit_in * (-Omega1_v /2/tf1 * (1 + 4.5 * ein_squared + 0.625 * ein_fourth) / one_minus_einsq**5)
            
            Y1 += 1.0/norbit_in * (Omega1_u /2/tf1 * (1 + 1.5 * ein_squared + 0.125 * ein_fourth) / one_minus_einsq**5)                    

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
    

    dS0x_dt = 0
    
    dS0y_dt = 0
    
    dS0z_dt = 0
    
    dS1x_dt = 0
    
    dS1y_dt = 0
    
    dS1z_dt = 0


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
        
    
        dS0x_dt += mu * hin * (- Y0 * uinx + X0 * vinx + W0 * ninx)
        
        dS0y_dt += mu * hin * (- Y0 * uiny + X0 * viny + W0 * niny)
        
        dS0z_dt += mu * hin * (- Y0 * uinz + X0 * vinz + W0 * ninz)
        
        dS1x_dt += mu * hin * (- Y1 * uinx + X1 * vinx + W1 * ninx)
        
        dS1y_dt += mu * hin * (- Y1 * uiny + X1 * viny + W1 * niny)
        
        dS1z_dt += mu * hin * (- Y1 * uinz + X1 * vinz + W1 * ninz)


        
    ########################################################################
    diffeq_list = [deinx_dt, 
                   deiny_dt, 
                   deinz_dt, 
                   dhinx_dt, 
                   dhiny_dt, 
                   dhinz_dt,
                   deoutx_dt, 
                   deouty_dt, 
                   deoutz_dt, 
                   dhoutx_dt, 
                   dhouty_dt, 
                   dhoutz_dt,
                   dS0x_dt,
                   dS0y_dt,
                   dS0z_dt,
                   dS1x_dt,
                   dS1y_dt,
                   dS1z_dt]

    return diffeq_list
