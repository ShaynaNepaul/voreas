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

def get_velocity_value(name, T, p, S1, S2, S3, S4): 

    gas_class = gaz_classes[name]
    a = gas_class(iso_type='IsoBar', temp=T, press=p) 
    velocity = a.get_velocity(T, p)
    return velocity

def get_density_value(name, T, p, S1, S2, S3, S4): 
    velocity = get_velocity_value(name, T, p, S1, S2, S3, S4)
    corr = None
    Z1 = None 
    Z2 = None 
    Z3 = None 
    Z4 = None
    for d in data:
        if d["name"] == name:
            corr = d["corr"]
            Z1 = d["S1_S2"]
            Z2 = d["S1_S2"]
            Z3 = d["S3_S4"]
            Z4 = d["S3_S4"]
    dx = 0.1 
    factor = corr * 4 / (np.pi * K_b * T * dx * velocity)
    sum = Z1 * S1 + Z2 * S2 + Z3 * S3 + Z4 * S4
    density = factor * sum
    return density

def main(): #test
    print(get_velocity_value(name = "H2", T = 40, p = 10, S1 = 2, S2 = 2, S3 = 5, S4 = 4))
    print(get_density_value(name = "H2", T = 40, p = 10, S1 = 2, S2 = 2, S3 = 5, S4 = 4))

if __name__ == "__main__":
    main()