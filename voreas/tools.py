from gas import * 

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate import UnivariateSpline
# Needed constants
# Unit mass of the gas species [kg]
M_u = 1.660538e-27

# Boltzmann constant [J/k]
K_b = 1.3806488e-23

def main():
    # test with H2 
    a = H2(iso_type='IsoBar', temp=9, t_low=18, t_high=50, press=8)

    result = a.velocity()
    fit_equations = a.fit_function()
    
    def get_velocity(t,p): 
        t = 45 #test
        if len(fit_equations) == 4 : 
            temp_start, phase_start, phase_end, temp_before_array, v_before, temp_after_array, v_after, x, y = result
            data_before_polynomial, m,b, data_after_polynomial = fit_equations
            
            if t <= temp_before_array[-1]: 
                return (data_before_polynomial(t))
            elif t>= temp_after_array[0] : 
                return (data_after_polynomial(t))
            elif t <= x[0] and t>= x[1]: 
                print(m*t + b)
        else : 
            data_polynomial = fit_equations
            return data_polynomial(t)
        
    def get_density(t,p): #,s,dx,species
        v = get_velocity(t, p)
        return v * 1e9
    
    density = get_density(45, 8)

if __name__ == "__main__":
    main()