

# Needed libraries
from __future__ import division
import numpy as np
from io import BytesIO
import os.path
from os import makedirs
from numpy.lib.scimath import sqrt as csqrt
from io import StringIO  
import pandas as pd 
from urllib.request import Request, urlopen
import math

# Needed constants
# Unit mass of the gas species [kg]
M_u = 1.660538e-27

# Boltzmann constant [J/k]
K_b = 1.3806488e-23


class Gas:
    def __init__(self, temp=0, press=0, iso_type='IsoBar', t_low=0, t_high=0, p_low=0, p_high=0):
        self.temp = temp
        self.press = press
        self.iso_type = iso_type
        self.t_low = t_low
        self.t_high = t_high
        self.p_low = p_low
        self.p_high = p_high
        self.long_name = ''
        self.short_name = ''
        self.kappa = 0
        self.mass = 0
        self.liquid_enthalpy_tp = 0
        self.vapour_enthalpy_tp = 0
        self.gas_id = ''
        self.local_directory = os.path.expanduser('~') + '/NIST-gas_database/'
        if self.iso_type == 'IsoBar':
            pass
        elif self.iso_type == 'IsoTherm':
            pass
        else:
            self.iso_type = 'IsoBar'

    # --------- Saving/loading data ---------

    def get_file_name(self):
        if self.iso_type == 'IsoBar':
            file_name = self.short_name + '_IsoBar_' + str(self.temp) + '_K_' + str(self.t_low) + '-' + str(
                self.t_high) + '_K_at_' + str(self.press) + '_bar'
        elif self.iso_type == 'IsoTherm':
            file_name = self.short_name + '_IsoTherm_' + str(self.press) + '_bar_' + str(self.p_low) + '-' + str(
                self.p_high) + '_bar_at_' + str(self.temp) + '_K'
        else:
            self.iso_type = 'IsoBar'
            file_name = self.short_name + '_IsoBar_' + str(self.temp) + '_K_' + str(
                self.t_low) + '-' + str(self.t_high) + '_K_at_' + str(self.press) + '_bar'
        return file_name

    #---------- Gas properties --------------

    def get_long_name(self):
        """The long name of the chosen gas species."""
        return self.long_name

    def get_short_name(self):
        """The short name of the chosen gas species."""
        return self.short_name
    
    def get_kappa(self): 
        """The adiabatic index [c_p/c_V] of the chosen gas species."""
        return self.kappa

    def get_mass(self):
        """The mass number of the gas particles (atoms / molecules)."""
        return self.mass

    def get_liquid_enthalpy_tp(self):  
        """The liquid enthalpy [kJ/kg] of the chosen gas species at its triple point."""
        return self.liquid_enthalpy_tp

    def get_vapour_enthalpy_tp(self):
        """The vapour enthalpy [kJ/kg] of the chosen gas species at its triple point."""
        return self.vapour_enthalpy_tp
    
    def get_mean_enthalpy_at_tp(self):
        """Calculate the mean enthalpy at the triple point of the gas."""
        mean_enthalpy = 0.5 * (self.get_liquid_enthalpy_tp() + self.get_vapour_enthalpy_tp())
        return mean_enthalpy

    def get_gas_id(self):
        """The individual gas ID number, found in http://webbook.nist.gov/chemistry/fluid/"""
        return self.gas_id

    # ---------------- Download data from NIST -----------------------

    def get_nist_data_url(self):
        """Define the URL where the raw data can be found."""
        url_default = 'http://webbook.nist.gov/cgi/fluid.cgi?Action=Data&Wide=on'
        url_units = '&Digits=5&RefState=DEF&TUnit=K&PUnit=bar&DUnit=kg%2Fm3&HUnit=kJ%2Fkg&WUnit=m%2Fs&VisUnit=Pa*s&STUnit=N%2Fm'
        if self.iso_type == 'IsoBar':
            url_nist = url_default + '&ID=' + self.gas_id + '&Type=' + self.iso_type + '&P=' + str(
                self.press) + '&THigh=' + str(self.t_high) + '&TLow=' + str(
                self.t_low) + '&TInc=0.001' + url_units
        elif self.iso_type == 'IsoTherm':
            url_nist = url_default + '&ID=' + self.gas_id + '&Type=' + self.iso_type + '&T=' + str(
                self.temp) + '&PHigh=' + str(self.p_high) + '&PLow=' + str(self.p_low) + '&PInc=0.001' + url_units
        else:
            self.iso_type = 'IsoBar'
            url_nist = url_default + '&ID=' + self.gas_id + '&Type=' + self.iso_type + '&P=' + str(
                self.press) + '&THigh=' + str(self.t_high) + '&TLow=' + str(
                self.t_low) + '&TInc=0.001' + url_units
        return url_nist

    def download_raw_nist_data(self):
        """Download the raw gas table data from the NIST database."""
        headers = {'User-Agent': 'Mozilla/5.0'}
        request = Request(self.get_nist_data_url(), headers=headers)
        with urlopen(request) as source_url:
            raw_nist_data = source_url.read()
        return raw_nist_data
    
    # -------------- Save downloaded data locally --------------

    def save_file(self):
        """Save the file to the defined location (self.local_directory)."""
        if os.path.isfile(self.local_directory + self.get_file_name()):
            pass
        else:
            with open(self.local_directory + self.get_file_name(), 'w') as data_file:
                data_file.write(self.download_raw_nist_data().decode(encoding='UTF-8'))
        return

    def save_gas_data(self):
        """Check the path and save the file."""
        if os.path.exists(self.local_directory):
            self.save_file()
        else:
            makedirs(self.local_directory)
            self.save_file()
        return

    def get_raw_nist_data(self):
        """Get the raw gas table data from the NIST database (local or: from the web - then save it!)."""
        if os.path.isfile(self.local_directory + self.get_file_name()):
            with open(self.local_directory + self.get_file_name(), 'r') as data_file:
                raw_nist_data = data_file.read().encode(encoding='UTF-8')
        else:
            raw_nist_data = self.download_raw_nist_data()
            self.save_gas_data()
        return raw_nist_data
    
    # -------------- Convert raw data to pandas DataFrame --------------

    def get_dataframe_data(self):
        """ Transforme bytes data in a Pandas Dataframe """
        brut_data = self.get_raw_nist_data()
        brut_data = brut_data.decode('utf-8')
        print(brut_data)
        data_stream = StringIO(brut_data)
        df = pd.read_csv(data_stream, delimiter="\t", header=None, skiprows=1)
        return df

    #-------------------- Get Dataframe column -----------------------------
    
    def get_dataframe_column(self, column_number):

        """ Access to one colomn by its number
            
            Return : Dataframe column 
        """
        df = self.get_dataframe_data()
        column_data = df[column_number]
        return column_data
    
    def get_phase_column(self):
        """ Returns list associated to the phase label """
        column = self.get_dataframe_column(13)
        Phase = column.astype(str).to_list()
        return Phase 
    
    def get_temperature_column(self): 
        """ Returns np.array associated to the temperature """
        column = self.get_dataframe_column(0)
        Temperature = column.astype(float).to_numpy()
        return Temperature
    
    def get_pressure_column(self): 
        """ Returns np.array associated to the pressure """
        column = self.get_dataframe_column(1)
        Pressure = column.astype(float).to_numpy()
        return Pressure

    def get_density_column(self): 
        """ Returns np.array associated to the density """
        column = self.get_dataframe_column(2)
        Density = column.astype(float).to_numpy()
        return Density

    def get_enthalpy_column(self): 
        """ Returns np.array associated to the enthalpy """
        column = self.get_dataframe_column(5)
        Enthalpy = column.astype(float).to_numpy()
        return Enthalpy

    def get_cv_column(self): 
        """ Returns np.array associated to cv """
        column = self.get_dataframe_column(7)
        cv = column.astype(float).to_numpy()
        return cv

    def get_cp_column(self): 
        """ Returns np.array associated to cp """
        column = self.get_dataframe_column(8)
        cp = column.astype(float).to_numpy()
        return cp

    # ----------------- Detect transition --------------------------

    def detect_transition(self):

        """ Detects a phase transition with label 

        Returns : 
        - index i-1
        - initial phase 
        - temperature at the start of the transition
        - pressure at the start of the transition
        """

        phase = self.get_phase_column()
        transition = []
        temperature = self.get_temperature_column()
        pressure = self.get_pressure_column()

        for i in range(1, len(phase)): 
            if phase[i] != phase[i-1] : 
                t_start = temperature[i-1]
                t_end = t_start + 0.15*t_start
                indices = np.where(temperature >= t_end)[0]
                index_T2_real = indices[0] if len(indices) > 0 else None
                transition_entry = [ i-1, phase[i-1], t_start, pressure[i-1], index_T2_real, phase[i], temperature[index_T2_real], pressure[index_T2_real] ]
                transition.append(transition_entry)

            else : 
                transition.append(None)

        return transition
    
    
    def get_phase_array(self, phase_name) : 

        """ Extracts data arrays corresponding to a specific phase 
        
        Returns:
        tuple:
            - transition (tuple or None): The first detected transition tuple if available and matching phase, otherwise None.
            - temperature (np.ndarray): Temperature values corresponding to the specified phase.
            - pressure (np.ndarray): Pressure values corresponding to the specified phase.
            - density (np.ndarray): Density values corresponding to the specified phase.
            - enthalpy (np.ndarray): Enthalpy values corresponding to the specified phase.
        """

        temperature = self.get_temperature_column()
        pressure = self.get_pressure_column()
        density = self.get_density_column()
        enthalpy = self.get_enthalpy_column()
        phase = self.get_phase_column()
        detect_transition = self.detect_transition()
        transitions = [t for t in detect_transition if t is not None]

        if len(transitions) > 0 : #if a transition occurs
            transition = transitions[0]
            index_start, phase_start, *_ = transition
            if phase_start == phase_name : 
                return transition, temperature[:index_start], pressure[:index_start], density[:index_start], enthalpy[:index_start]
            else : 
                mask = np.array(self.get_phase_column()) == phase_name
                return None, temperature[mask], pressure[mask], density[mask], enthalpy[mask]
            
    def get_liq_velocity_array(self, press_liq_array, density_liq_array) : 
        """ Calculate liquid velocity using Bernoulli formula for a given array of pressure and density"""
        v_liq_array = csqrt((2 * 1e5 * press_liq_array) / density_liq_array)
        return v_liq_array
        
    def get_gas_velocity_array(self, temp_gas_array) : 
        """ Calculate gas velocity using ideal gas model for a given temperature """
        v_gas_array = csqrt(
            ((2 * self.kappa) / (self.kappa - 1)) * (K_b / (M_u * self.mass)) * temp_gas_array)
        return v_gas_array
    
    def get_sc_velocity_array(self, enthalpy_sc_array) : 
        """ Calculate supercritical fluid velocity using the simple enthalpy model """
        v_array_calculation = csqrt(2 * 1000 * (enthalpy_sc_array - self.get_mean_enthalpy_at_tp()))
        v_array = np.where(np.iscomplex(v_array_calculation), 0, v_array_calculation).astype(float)
        return v_array
            
    def calculate_velocity(self, phase, temp_array, press_array=None, density_array=None, enthalpy_array=None) : 
        """ Calculate velocity based on the specified phase and corresponding thermodynamic data """
        if phase == 'liquid': 
            return self.get_liq_velocity_array(press_array, density_array)
        elif phase == 'supercritical' : 
            return self.get_sc_velocity_array(enthalpy_array)
        else : 
            return self.get_gas_velocity_array(temp_array)
        
    def velocity(self): 

        """
        Compute velocity profiles across phase regions and transitions.

        If a phase transition is detected : 
        It calculates:
        - Velocity before the transition
        - Velocity after the transition
        - A linear interpolation across the transition temperature range
        It returns : 
        tuple:
                - temp_start (float): Starting temperature of the transition.
                - phase_start (str): Initial phase before the transition.
                - phase_end (str): Final phase after the transition.
                - temp_before (np.ndarray): Temperature array before the transition.
                - v_before (np.ndarray): Velocity array before the transition.
                - temp_after (np.ndarray): Temperature array after the transition.
                - v_after (np.ndarray): Velocity array after the transition.
                - x (np.ndarray): Two-point array for temperature range during transition.
                - y (np.ndarray): Two-point array for velocity values during transition.

        Otherwise, it returns : 
        tuple: 
                - temp_array (np.ndarray): Temperature array for the current phase.
                - v (np.ndarray): Velocity array for the current phase.
                - phase (str): Name of the phase.
        """

        phases = [ 'liquid', 'supercritical', 'vapor']

        for phase in phases : 
            phase_data = self.get_phase_array(phase)
            if phase_data[0] is None : #No transition
                 _, temp_array, press_array, density_array, enthalpy_array = phase_data
                 v = self.calculate_velocity(phase, temp_array, press_array, density_array, enthalpy_array)
                 return temp_array, v, phase
            else : 
                transition, temp_before, press_before, density_before, enthalpy_before = phase_data
                index_start, phase_start, temp_start, press_start, index_end, phase_end, temp_end, press_end = transition

                # Velocity before transition
                v_before = self.calculate_velocity(phase_start, temp_before, press_before, density_before, enthalpy_before)

                # Data after transition
                temperature = self.get_temperature_column()
                pressure = self.get_pressure_column()
                density = self.get_density_column()
                enthalpy = self.get_enthalpy_column()

                temp_after = temperature[index_end:]

                if phase_end == 'liquid':
                    press_after = pressure[index_end:]
                    density_after = density[index_end:]
                    enthalpy_after = None
                elif phase_end == 'supercritical':
                    press_after = None
                    density_after = None
                    enthalpy_after = enthalpy[index_end:]
                elif phase_end == 'vapor':
                    press_after = None
                    density_after = None
                    enthalpy_after = None
                else:
                    raise ValueError(f"Phase finale inconnue : {phase_end}")

                v_after = self.calculate_velocity(phase_end, temp_after, press_after, density_after, enthalpy_after)

                # During transition
                T1, T2 = temp_start, temperature[index_end]
                v1, v2 = v_before[-1], v_after[0]
                x = np.array([T1, T2])
                y = np.array([v1, v2])

                return temp_start, phase_start, phase_end, temp_before, v_before, temp_after, v_after, x, y               

    def fit_function(self): 

        """
    Fit polynomial functions to velocity data across different phases.

    Polynomial models of degree 3 to the data depending on whether a 
    phase transition is present.

    Returns:
        If no phase transition is detected:
            np.poly1d: A 3rd-degree polynomial fitted to the entire (x, y) dataset.

        If a phase transition is detected:
            tuple:
                - np.poly1d: Polynomial for the 'before' phase segment.
                - float: Slope (m) of the linear fit during the transition.
                - float: Intercept (b) of the linear fit during the transition.
                - np.poly1d: Polynomial for the 'after' phase segment.
    """

        result = self.velocity()
        if len(result) == 3 :
            x, y, phase = result
            
            coeff = np.polyfit(x, y, 3)
            data_polynomial = np.poly1d(coeff)

            return data_polynomial
        
        else : 
            temp_start, phase_start, phase_end, temp_before_array, v_before, temp_after_array, v_after, x, y = result

            x_before = temp_before_array
            y_before = v_before
            coeff_before = np.polyfit(x_before, y_before, 3)
            data_before_polynomial = np.poly1d(coeff_before)

            x_during = x 
            y_during = y 
            m = (y_during[1] - y_during[0]) / (x_during[1] - x_during[0])
            b =  y_during[0] - m * x_during[0]

            x_after = temp_after_array
            y_after = v_after
            coeff_after = np.polyfit(x_after, y_after, 3)
            data_after_polynomial = np.poly1d(coeff_after)

            return data_before_polynomial, m,b, data_after_polynomial
        
    def get_velocity(self, t, p): 

        """ Calculate the velocity for any temperature based on the fit. 
        This method retrieves temperature and velocity data using 'fit function()' 

        Return float : value of the velocity for a given temperature and pressure
        
        """
        result = self.velocity()
        fit_equations = self.fit_function()

        if len(fit_equations) == 4 : 
            temp_start, phase_start, phase_end, temp_before_array, v_before, temp_after_array, v_after, x, y = result
            data_before_polynomial, m,b, data_after_polynomial = fit_equations
            """
            if t == temp_before_array[-1] or t == x[0] or t == x[1] or t == temp_after_array[0]:
                print(f"Edge temperature: {t}")
                print(f"temp_before_array[-1]: {temp_before_array[-1]}")
                print(f"x: {x}, temp_after_array[0]: {temp_after_array[0]}")
            """
            
            if t <= temp_before_array[-1]: 
                return (data_before_polynomial(t))
            elif t>= temp_after_array[0] : 
                return (data_after_polynomial(t))
            elif x[0] <= t <= x[1]: 
                return m*t + b
            elif temp_before_array[-1] <= t <= x[0]: 
                return data_before_polynomial(t)
            else : 
                print("Error")
        else : 
            data_polynomial = fit_equations
            return data_polynomial(t)

class N2(Gas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.temp == 0:
            self.temp = 140
        if self.iso_type == 'IsoBar':
            if self.t_low == 0:
                self.t_low = math.ceil(63.15 * 10) / 10
            if self.t_high == 0:
                self.t_high = 300
        if self.press == 0:
            self.press = 10
        if self.iso_type == 'IsoTherm':
            if self.p_low == 0:
                self.p_low = self.press - 1
            if self.p_high == 0:
                self.p_high = self.press + 1
        self.long_name = 'Nitrogen'
        self.short_name = 'N2'
        self.kappa = 1.4
        self.mass = 2 * 14.0067
        self.liquid_enthalpy_tp = -150.75
        self.vapour_enthalpy_tp = 46.211
        self.gas_id = 'C7727379'


class H2(Gas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.temp == 0:
            self.temp = 40
        if self.iso_type == 'IsoBar':
            if self.t_low == 0:
                self.t_low = math.ceil(13.957 * 10) / 10
            if self.t_high == 0:
                self.t_high = 300
        if self.press == 0:
            self.press = 40
        if self.iso_type == 'IsoTherm':
            if self.p_low == 0:
                self.p_low = self.press - 1
            if self.p_high == 0:
                self.p_high = self.press + 1
        self.long_name = 'Hydrogen'
        self.short_name = 'H2'
        self.kappa = 1.41
        self.mass = 2 * 1.008
        self.liquid_enthalpy_tp = -53.923
        self.vapour_enthalpy_tp = 399.82
        self.gas_id = 'C1333740'


class He(Gas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.temp == 0:
            self.temp = 10
        if self.iso_type == 'IsoBar':
            if self.t_low == 0:
                self.t_low = 2.3
            if self.t_high == 0:
                self.t_high = 300
        if self.press == 0:
            self.press = 5
        if self.iso_type == 'IsoTherm':
            if self.p_low == 0:
                self.p_low = self.press - 1
            if self.p_high == 0:
                self.p_high = self.press + 1
        self.long_name = 'Helium'
        self.short_name = 'He'
        self.kappa = 1.67
        self.mass = 4.002602
        self.liquid_enthalpy_tp = -7.5015
        self.vapour_enthalpy_tp = 15.726
        self.gas_id = 'C7440597'


class Ne(Gas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.temp == 0:
            self.temp = 120
        if self.iso_type == 'IsoBar':
            if self.t_low == 0:
                self.t_low = math.ceil(24.5561 * 10) / 10
            if self.t_high == 0:
                self.t_high = 300
        if self.press == 0:
            self.press = 10
        if self.iso_type == 'IsoTherm':
            if self.p_low == 0:
                self.p_low = self.press - 1
            if self.p_high == 0:
                self.p_high = self.press + 1
        self.long_name = 'Neon'
        self.short_name = 'Ne'
        self.kappa = 1.67
        self.mass = 20.1797
        self.liquid_enthalpy_tp = -4.9151
        self.vapour_enthalpy_tp = 83.204
        self.gas_id = 'C7440019'


class Ar(Gas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.temp == 0:
            self.temp = 180
        if self.iso_type == 'IsoBar':
            if self.t_low == 0:
                self.t_low = math.ceil(83.78 * 10) / 10
            if self.t_high == 0:
                self.t_high = 300
        if self.press == 0:
            self.press = 10
        if self.iso_type == 'IsoTherm':
            if self.p_low == 0:
                self.p_low = self.press - 1
            if self.p_high == 0:
                self.p_high = self.press + 1
        self.long_name = 'Argon'
        self.short_name = 'Ar'
        self.kappa = 1.67
        self.mass = 39.948
        self.liquid_enthalpy_tp = -121.44
        self.vapour_enthalpy_tp = 42.281
        self.gas_id = 'C7440371'


class Kr(Gas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.temp == 0:
            self.temp = 300
        if self.iso_type == 'IsoBar':
            if self.t_low == 0:
                self.t_low = math.ceil(115.763 * 10) / 10
            if self.t_high == 0:
                self.t_high = 300
        if self.press == 0:
            self.press = 10
        if self.iso_type == 'IsoTherm':
            if self.p_low == 0:
                self.p_low = self.press - 1
            if self.p_high == 0:
                self.p_high = self.press + 1
        self.long_name = 'Krypton'
        self.short_name = 'Kr'
        self.kappa = 1.67
        self.mass = 83.798
        self.liquid_enthalpy_tp = -2.0639
        self.vapour_enthalpy_tp = 106.43
        self.gas_id = 'C7439909'


class Xe(Gas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.temp == 0:
            self.temp = 300
        if self.iso_type == 'IsoBar':
            if self.t_low == 0:
                self.t_low = 161.391
            if self.t_high == 0:
                self.t_high = 300
        if self.press == 0:
            self.press = 10
        if self.iso_type == 'IsoTherm':
            if self.p_low == 0:
                self.p_low = self.press - 1
            if self.p_high == 0:
                self.p_high = self.press + 1
        self.long_name = 'Xenon'
        self.short_name = 'Xe'
        self.kappa = 1.67
        self.mass = 131.293
        self.liquid_enthalpy_tp = -1.2418
        self.vapour_enthalpy_tp = 95.165
        self.gas_id = 'C7440633'


class Ubertest:
    def __init__(self):
        self.long_name = ''
        self.kappa = 0
        pass

    def get_long_name(self):
        return self.long_name

    def get_kappa(self):
        return self.kappa


class Test1(Ubertest):
    def __init__(self):
        super().__init__()
        self.long_name = 'Helium'
        self.kappa = 23

    @staticmethod
    def get_kapp(x):
        return x.kappa


class Test2(Ubertest):
    def __init__(self):
        super().__init__()
        self.long_name = 'Hydrogen'
        self.kappa = 66


def get_k(x):
    return x.kappa
