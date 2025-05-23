from gas import *  
import numpy as np
import matplotlib.pyplot as plt

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

data = [ #values not found for Ne, Kr, Xe
    {"name": "H2", "corr": 2.4, "S1_S2": 555, "S3_S4" : 220 },
    {"name": "N2", "corr": 1, "S1_S2": 685, "S3_S4" : 260},
    {"name": "He", "corr": 5.9, "S1_S2": 655, "S3_S4" : 255},
    {"name": "Ne", "corr": 3.33, "S1_S2": 600, "S3_S4" : 200 },
    {"name": "Ar", "corr": 0.78, "S1_S2": 665, "S3_S4" : 255},
    {"name": "Kr", "corr": 0.5, "S1_S2": 600, "S3_S4" : 200},
    {"name": "Xe", "corr": 0.85, "S1_S2": 600, "S3_S4" : 200}
]


def plot_velocity(name, T, p, S1, S2, S3, S4): #fixed pressure, fixed temperature

    """ Calculate the velocity and the density
    
    S1, S2, S3, S4 : correspond to the pressure at the differential pumping stages 
    Z1, Z2, Z3, Z4 : correspond to pumping speed of each pump 

    """

    gas_class = gaz_classes[name]
    a = gas_class(iso_type='IsoBar', temp=T, press=p) 
    velocity = a.get_velocity(T, p)
    result = a.velocity()

    
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

def main(): #test
    print(plot_velocity(name = "H2", T = 40, p = 10, S1 = 2, S2 = 2, S3 = 5, S4 = 4))

if __name__ == "__main__":
    main()