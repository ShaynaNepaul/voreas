#!/usr/bin/env python

# Needed libraries

from urllib.request import urlopen
from numpy import sqrt
import numpy as np
from io import BytesIO
import os.path
from os import makedirs
from numpy.lib.scimath import sqrt as csqrt
from io import StringIO  
import pandas as pd 
from urllib.request import Request, urlopen

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

    # --------- IMPORT NIST DATA BETWEEN t_low and t_high ---------

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

    def get_long_name(self):
        """The long name of the chosen gas species."""
        return self.long_name

    def get_short_name(self):
        """The short name of the chosen gas species."""
        return self.short_name
    
    def get_kappa(self): #constant 
        """The adiabatic index [c_p/c_V] of the chosen gas species."""
        return self.kappa

    def get_mass(self):#constant 
        """The mass number of the gas particles (atoms / molecules)."""
        return self.mass

    def get_liquid_enthalpy_tp(self): #constant 
        """The liquid enthalpy [kJ/kg] of the chosen gas species at its triple point."""
        return self.liquid_enthalpy_tp

    def get_vapour_enthalpy_tp(self): #constant 
        """The vapour enthalpy [kJ/kg] of the chosen gas species at its triple point."""
        return self.vapour_enthalpy_tp
    
    def get_mean_enthalpy_at_tp(self):
        """Calculate the mean enthalpy at the triple point of the gas."""
        mean_enthalpy = 0.5 * (self.get_liquid_enthalpy_tp() + self.get_vapour_enthalpy_tp())
        return mean_enthalpy

    def get_gas_id(self):
        """The individual gas ID number, found in http://webbook.nist.gov/chemistry/fluid/"""
        return self.gas_id

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
        """Download the raw gas table data from the NIST database.""" #problem with the url fixed
        headers = {'User-Agent': 'Mozilla/5.0'}
        request = Request(self.get_nist_data_url(), headers=headers)
        with urlopen(request) as source_url:
            raw_nist_data = source_url.read()
        return raw_nist_data
    
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
        #print(raw_nist_data) #ok
        return raw_nist_data
    
    def get_dataframe_data(self): #added

        """ Transforme bytes data in a Pandas Dataframe """
        brut_data = self.get_raw_nist_data()
        brut_data = brut_data.decode('utf-8')
        data_stream = StringIO(brut_data)
        df = pd.read_csv(data_stream, delimiter="\t", header=None, skiprows=1)
        return df
    
    
    """ Added a csv file to write down our data ?? 

    def get_csv_file(self): 
    
    """

    #------- Get Dataframe column -------------------
    
    def get_dataframe_column(self, column_number): #added
        """ Access to one colomn by its number

        Return : Dataframe column 
        
        """
        df = self.get_dataframe_data()
        column_data = df[column_number]
        return column_data
    
    def get_phase_column(self): #added 
        """ Returns list associated to the phase label"""
        column = self.get_dataframe_column(13)
        Phase = column.astype(str).to_list()
        return Phase 
    
    def get_temperature_column(self): #modified
        """ Returns np.array associated to the temperature"""
        column = self.get_dataframe_column(0)
        Temperature = column.astype(float).to_numpy()
        return Temperature
    
    def get_pressure_column(self): #modified
        """ Returns np.array associated to the pressure"""
        column = self.get_dataframe_column(1)
        Pressure = column.astype(float).to_numpy()
        return Pressure

    def get_density_column(self): #modified
        """ Returns np.array associated to the density"""
        column = self.get_dataframe_column(2)
        Density = column.astype(float).to_numpy()
        return Density

    def get_enthalpy_column(self): #modified
        """ Returns np.array associated to the enthalpy"""
        column = self.get_dataframe_column(5)
        Enthalpy = column.astype(float).to_numpy()
        return Enthalpy

    def get_cv_column(self): #modified
        """ Returns np.array associated to cv"""
        column = self.get_dataframe_column(7)
        cv = column.astype(float).to_numpy()
        return cv

    def get_cp_column(self): #modified
        """ Returns np.array associated to cp"""
        column = self.get_dataframe_column(8)
        cp = column.astype(float).to_numpy()
        return cp

    
    # --------- Detect transition --------------
    
    def detect_transition(self): #added

        """ Detect transition with label (first approach) and return :
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
    
    #-------- Data selection  : Liquid, Supercritical, Gas according to phase labels -----------

    """ Security function ?? """
    
    def get_liquid_array(self) : 

        " Extract liq array according to the label 'liq' until the transition occurs if its the case. Otherwise, get the whole array"

        temperature = self.get_temperature_column()
        pressure = self.get_pressure_column()
        density = self.get_density_column()
        enthalpy = self.get_enthalpy_column()
        phase = self.get_phase_column()
        detect_transition = self.detect_transition()
        transitions = [t for t in detect_transition if t is not None]

        if len(transitions) > 0 : 
            #First transition
            transition = transitions[0]
            index_start, phase_start, temp_start, press_start, index_end, phase_end, temp_end, press_end = transition

            if phase_start == 'liquid' : 
                temp_liq_array = temperature[:index_start] #until the transition occurs
                press_liq_array = pressure[:index_start]
                density_liq_array = density[:index_start]
                enthalpy_liq_array = enthalpy[:index_start]


            return transition, temp_liq_array, press_liq_array, density_liq_array, enthalpy_liq_array
        
        else : 
            
            phase = np.array(self.get_phase_column(), dtype=str)
            mask_liquid = phase == 'liquid'
            temp_liq_array = temperature[mask_liquid] #take the whole array indicated by 'liquid' label in phase column 
            press_liq_array = pressure[mask_liquid]
            density_liq_array = density[mask_liquid]
            enthalpy_liq_array = enthalpy[mask_liquid]

            return temp_liq_array, press_liq_array, density_liq_array, enthalpy_liq_array
        
    def get_supercritical_array(self) : 

        " Extract sc array according to the label 'liq' until the transition occurs if its the case. Otherwise, get the whole array"

        temperature = self.get_temperature_column()
        pressure = self.get_pressure_column()
        density = self.get_density_column()
        enthalpy = self.get_enthalpy_column()
        phase = self.get_phase_column()
        detect_transition = self.detect_transition()
        transitions = [t for t in detect_transition if t is not None]
        
        if len(transitions) > 0 : 

            #First transition
            transition = transitions[0]
            index_start, phase_start, temp_start, press_start, index_end, phase_end, temp_end, press_end = transition

            if phase_start == 'supercritical' : 
                temp_sc_array = temperature[:index_start] #until the transition occurs
                press_sc_array = pressure[:index_start]
                density_sc_array = density[:index_start]
                enthalpy_sc_array = enthalpy[:index_start]

                return transition, temp_sc_array, press_sc_array, density_sc_array, enthalpy_sc_array
        
        else : 

            phase = np.array(self.get_phase_column(), dtype=str)
            mask_sc = phase == 'supercritical'
            temp_sc_array = temperature[mask_sc] #take the whole array indicated by 'liquid' label in phase column 
            press_sc_array = pressure[mask_sc]
            density_sc_array = density[mask_sc]
            enthalpy_sc_array = enthalpy[mask_sc]

            return temp_sc_array, press_sc_array, density_sc_array, enthalpy_sc_array
        
    def get_gas_array(self) : 

        " Extract sc array according to the label 'liq' until the transition occurs if its the case. Otherwise, get the whole array"

        temperature = self.get_temperature_column()
        pressure = self.get_pressure_column()
        density = self.get_density_column()
        enthalpy = self.get_enthalpy_column()
        phase = self.get_phase_column()
        detect_transition = self.detect_transition()
        transitions = [t for t in detect_transition if t is not None]
        
        if len(transitions) > 0 : 

            #First transition
            transition = transitions[0]
            index_start, phase_start, temp_start, press_start, index_end, phase_end, temp_end, press_end = transition

            if phase_start == 'vapor' : 
                temp_gas_array = temperature[:index_start] #until the transition occurs
                press_gas_array = pressure[:index_start]
                density_gas_array = density[:index_start]
                enthalpy_gas_array = enthalpy[:index_start]

                return transition, temp_gas_array, press_gas_array, density_gas_array, enthalpy_gas_array
        
        else : 

            phase = np.array(self.get_phase_column(), dtype=str)
            mask_gas = phase == 'vapor'
            temp_gas_array = temperature[mask_gas] #take the whole array indicated by 'liquid' label in phase column 
            press_gas_array = pressure[mask_gas]
            density_gas_array = density[mask_gas]
            enthalpy_gas_array = enthalpy[mask_gas]

            return temp_gas_array, press_gas_array, density_gas_array, enthalpy_gas_array
        
        
    #------- Formula to calculate velocity : Update, according to the array entry not for the whole temperature array ------------

    def get_liq_velocity_array(self, press_liq_array, density_liq_array) : 
        """ Calculate liquid velocity using Bernoulli formulat for a given array of pressure and density"""
        v_liq_array = sqrt((2 * 1e5 * press_liq_array) / density_liq_array)
        return v_liq_array
        
    def get_gas_velocity_array(self, temp_gas_array) : 
        v_gas_array = sqrt(
            ((2 * self.kappa) / (self.kappa - 1)) * (K_b / (M_u * self.mass)) * temp_gas_array)
        return v_gas_array
    
    def get_sc_velocity_array(self, enthalpy_sc_array) : 
        v_array_calculation = csqrt(2 * 1000 * (enthalpy_sc_array - self.get_mean_enthalpy_at_tp()))
        v_array = np.where(np.iscomplex(v_array_calculation), 0, v_array_calculation).astype(float)
        return v_array
    
    # ----------- Transitions --------------

    def velocity(self): 
        liq_array = self.get_liquid_array()
        sc_array = self.get_supercritical_array()
        gas_array = self.get_gas_array()
        temperature = self.get_temperature_column()
        pressure = self.get_pressure_column()
        enthalpy = self.get_enthalpy_column()
        density = self.get_density_column()

        if liq_array is not None and len(liq_array) == 4 : #no transition
            temp_liq_array, press_liq_array, density_liq_array, enthalpy_liq_array = liq_array
            v_liq = self.get_liq_velocity_array(press_liq_array,density_liq_array)
            str_phase = 'liquid'
            return temp_liq_array, v_liq, str_phase
        
        elif sc_array is not None and len(sc_array) == 4 : 
            temp_sc_array, press_sc_array, density_sc_array, enthalpy_sc_array = sc_array
            v_sc = self.get_sc_velocity_array(enthalpy_sc_array)
            str_phase = 'supercritical'
            return temp_sc_array, v_sc, str_phase
        
        elif gas_array is not None and len(gas_array) == 4 : 
            temp_gas_array, press_gas_array, density_gas_array, enthalpy_gas_array = gas_array
            v_gas = self.get_gas_velocity_array(temp_gas_array)
            str_phase = 'gas'
            return temp_gas_array, v_gas, str_phase

        elif len(liq_array) == 5 : #if liq transition
            transition, temp_liq_array, press_liq_array, density_liq_array, enthalpy_liq_array = liq_array
            index_start, phase_start, temp_start, press_start, index_end, phase_end, temp_end, press_end = transition

            # --- Before transition -----
            v_liq_before = self.get_liq_velocity_array(press_liq_array,density_liq_array)
            
            #-- After transition ---
            if phase_end == 'vapor': 
                temp_liq_after_array = temperature[index_end:]
                v_liq_after = self.get_gas_velocity_array(temp_liq_after_array)

            elif phase_end == 'supercritical': 
                enthalpy_liq_after_array = enthalpy[index_end:]
                temp_liq_after_array = temperature[index_end:]
                v_liq_after = self.get_sc_velocity_array(enthalpy_liq_after_array) 

            #--- During transition ----

            T1 = temp_start
            T2 = temperature[index_end]

            v1 = v_liq_before[-1]
            v2 = v_liq_after[0]

            x = np.array([T1, T2])
            y = np.array([v1, v2])
            """
            # Créer des points x plus denses pour lisser la courbe
            x_smooth = np.linspace(T1, T2, 100)

            # Créer la spline (ordre 1 ici car seulement 2 points, ordre 3 pas possible)
            spline = make_interp_spline(x, y, k=1)  # k=1 = linéaire ; tu peux mettre k=2 si tu ajoutes un 3e point

            # Interpolation lissée
            y_smooth = spline(x_smooth)

            # Créer des points x plus denses pour lisser la courbe
            x_smooth = np.linspace(temp_start, temp_end, 100)

            # Créer la spline (ordre 1 ici car seulement 2 points, ordre 3 pas possible)
            spline = make_interp_spline(x, y, k=1)  # k=1 = linéaire ; tu peux mettre k=2 si tu ajoutes un 3e point

            # Interpolation lissée
            y_smooth = spline(x_smooth)
            """
           
            return temp_start, phase_start, phase_end, temp_liq_array, v_liq_before, temp_liq_after_array, v_liq_after, x, y


        elif len(gas_array) == 5 : #if gas transition
            transition, temp_gas_array, press_gas_array, density_gas_array, enthalpy_gas_array = gas_array
            index_start, phase_start, temp_start, press_start, index_end, phase_end, temp_end, press_end = transition

            # --- Before transition -----
            v_gas_before = self.get_liq_velocity_array(temp_gas_array)
            
            #-- After transition ---
            if phase_end == 'liquid': 
                pressure_gas_after = pressure[index_end:]
                density_gas_after = density[index_end:]
                temp_gas_after_array = temperature[index_end:]
                v_gas_after = self.get_liq_velocity_array(pressure_gas_after, density_gas_after)

            elif phase_end == 'supercritical': 
                enthalpy_liq_after_array = enthalpy[index_end:]
                temp_gas_after_array = temperature[index_end:]
                v_liq_after = self.get_sc_velocity_array(enthalpy_liq_after_array) 

            #--- During transition ----

            T1 = temp_start
            T2 = temperature[index_end]

            v1 = v_liq_before[-1]
            v2 = v_liq_after[0]

            x = np.array([T1, T2])
            y = np.array([v1, v2])

            """
            # Créer des points x plus denses pour lisser la courbe
            x_smooth = np.linspace(T1, T2, 100)

            # Créer la spline (ordre 1 ici car seulement 2 points, ordre 3 pas possible)
            spline = make_interp_spline(x, y, k=1)  # k=1 = linéaire ; tu peux mettre k=2 si tu ajoutes un 3e point

            # Interpolation lissée
            y_smooth = spline(x_smooth)

            # Créer des points x plus denses pour lisser la courbe
            x_smooth = np.linspace(temp_start, temp_end, 100)

            # Créer la spline (ordre 1 ici car seulement 2 points, ordre 3 pas possible)
            spline = make_interp_spline(x, y, k=1)  # k=1 = linéaire ; tu peux mettre k=2 si tu ajoutes un 3e point

            # Interpolation lissée
            y_smooth = spline(x_smooth)
            """
           
            return temp_start, phase_start, phase_end, temp_gas_array, v_gas_before, temp_gas_after_array, v_gas_after, x, y
        
        elif len(sc_array) == 5 : #if gas transition
            transition, temp_sc_array, press_sc_array, density_sc_array, enthalpy_sc_array = sc_array
            index_start, phase_start, temp_start, press_start, index_end, phase_end, temp_end, press_end = transition

            # --- Before transition -----
            v_sc_before = self.get_sc_velocity_array(enthalpy_sc_array)
            
            #-- After transition ---
            if phase_end == 'liquid': 
                pressure_sc_after = pressure[index_end:]
                density_sc_after = density[index_end:]
                temp_sc_after_array = temperature[index_end:]
                v_sc_after = self.get_liq_velocity_array(pressure_sc_after, density_sc_after)

            elif phase_end == 'vapor': 
                temp_sc_after_array = temperature[index_end:]
                v_sc_after = self.get_gas_velocity_array(temp_sc_after_array) 

            #--- During transition ----

            T1 = temp_start
            T2 = temperature[index_end]

            v1 = v_liq_before[-1]
            v2 = v_liq_after[0]

            x = np.array([T1, T2])
            y = np.array([v1, v2])

            """
            # Créer des points x plus denses pour lisser la courbe
            x_smooth = np.linspace(T1, T2, 100)

            # Créer la spline (ordre 1 ici car seulement 2 points, ordre 3 pas possible)
            spline = make_interp_spline(x, y, k=1)  # k=1 = linéaire ; tu peux mettre k=2 si tu ajoutes un 3e point

            # Interpolation lissée
            y_smooth = spline(x_smooth)

            # Créer des points x plus denses pour lisser la courbe
            x_smooth = np.linspace(temp_start, temp_end, 100)

            # Créer la spline (ordre 1 ici car seulement 2 points, ordre 3 pas possible)
            spline = make_interp_spline(x, y, k=1)  # k=1 = linéaire ; tu peux mettre k=2 si tu ajoutes un 3e point

            # Interpolation lissée
            y_smooth = spline(x_smooth)
            """
           
            return temp_start, phase_start, phase_end, temp_sc_array, v_sc_before, temp_sc_after_array, v_sc_after, x, y
        
    def fit_function(self): 

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

            """
            plt.plot(temp_before_array,v_before, label = phase_start)
            plt.plot(temp_after_array, v_after, label = phase_end )
            plt.plot(x,y, "r--" )
            plt.axvline(x=temp_start, linestyle='--', color='black')
            plt.legend()
            plt.show()
            """


class N2(Gas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.temp == 0:
            self.temp = 140
        if self.iso_type == 'IsoBar':
            if self.t_low == 0:
                self.t_low = self.temp - 1
            if self.t_high == 0:
                self.t_high = self.temp + 1
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
                self.t_low = self.temp - 1
            if self.t_high == 0:
                self.t_high = self.temp + 1
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
                self.t_low = self.temp - 1
            if self.t_high == 0:
                self.t_high = self.temp + 1
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
                self.t_low = self.temp - 1
            if self.t_high == 0:
                self.t_high = self.temp + 1
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
                self.t_low = self.temp - 1
            if self.t_high == 0:
                self.t_high = self.temp + 1
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
                self.t_low = self.temp - 1
            if self.t_high == 0:
                self.t_high = self.temp + 1
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
                self.t_low = self.temp - 1
            if self.t_high == 0:
                self.t_high = self.temp + 1
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
