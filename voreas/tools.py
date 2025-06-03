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
    dx = 2 * 1e-3 # In meter
    factor = corr * 4 / (np.pi * K_b * T * dx * velocity)
    sum = (Z1* 1e-3) *(S1*1e2) + (Z2* 1e-3) *(S2*1e2) + (Z3* 1e-3) *(S3*1e2) + (Z4* 1e-3) *(S4*1e2)
    density = factor * sum
    return density / 10000 #in /cm2

def main(): #test
    T_rnge = np.linspace(20, 50, 100)
    #for t in T_range : 
        #print("temperature")
        #print(t)
        #print(f"{get_density_value(name='H2', T=t, p=6, S1=1.37e-7, S2=4.74e-9, S3=4.91e-9, S4=7.94e-10):.2e}")
    
    print(f"{get_density_value(name='H2', T= 38 , p=6, S1=1.37e-7, S2=4.74e-9, S3=4.91e-9, S4=7.94e-10):.2e}")

if __name__ == "__main__":
    main()