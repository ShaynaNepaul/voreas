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

gaz_classes = {
    "H2": H2,
    "He": He,
    "Ne": Ne,
    "Ar": Ar,
    "Kr": Kr,
    "Xe": Xe,
    "N2": N2
}

data = [
    {"name": "H2", "corr": 2.4, "speed": 1050},
    {"name": "N2", "corr": 1, "speed": 1100},
    {"name": "He", "corr": 5.9, "speed": 1320},
    {"name": "Ne", "corr": 3.33, "speed": 1200},
    {"name": "Ar", "corr": 0.78, "speed": 1000},
    {"name": "Kr", "corr": 0.5, "speed": 880},
    {"name": "Xe", "corr": 0.85, "speed": 880}
]

def print_velocity_density(name, T, p, dx=0.1, S=2): #fixed pressure

    gas_class = gaz_classes[name]
    a = gas_class(iso_type='IsoBar', temp=T, press=p) 
    velocity = a.get_velocity(T, p)
    result = a.velocity()

    # ----Visualize curves------
    
    if len(result) == 3:
        x, y, phase = result
        plt.plot(x, y, label=phase)
        plt.title(f"{name}: Velocity vs Temperature at {p} bar")
        plt.xlabel("Temperature [K]")
        plt.ylabel("Velocity [m/s]")
        plt.legend()
        plt.show()
    else:
        (temp_start, phase_start, phase_end,
         temp_before_array, v_before,
         temp_after_array, v_after, x, y) = result

        plt.plot(temp_before_array, v_before, label=phase_start)
        plt.plot(temp_after_array, v_after, label=phase_end)
        plt.plot(x, y, "r--", label="Interpolation")
        plt.axvline(x=temp_start, linestyle='--', color='black', label="Transition")
        plt.title(f"{name}: Velocity vs Temperature at {p} bar")
        plt.xlabel("Temperature [K]")
        plt.ylabel("Velocity [m/s]")
        plt.legend()
        plt.show()

    corr = None
    for d in data:
        if d["name"] == name:
            corr = d["corr"]

    p1, p2, p3, p4 = 1e-7, 1e-9, 1e-10, 1e-10 #fixed values
    delta_p = p1 + p2 + p3 + p4
    factor = 4 / (np.pi * K_b * T * dx * velocity)
    corrected_density = factor * corr * S * delta_p

    return {"velocity": velocity, "density": corrected_density} 

def get_velocity_value(name, T, p, dx=0.1, S=2): 
    data = print_velocity_density(name, T,p)
    v = data["velocity"]
    return v 

def get_density_value(name, T, p, dx=0.1, S=2): 
    data = print_velocity_density(name, T, p)
    d = data["density"]
    return d 

def main():
    T = 40 # temperature
    print(print_velocity_density(name="Ne", T=T, p = 10))
    print(get_velocity_value(name = "Ne", T = T, p = 10))
    print(get_density_value(name = "Ne", T = T, p = 10))

if __name__ == "__main__":
    main()