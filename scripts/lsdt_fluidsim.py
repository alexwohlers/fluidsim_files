#!/usr/bin/env python
# coding: utf-8

"""
Stand 16.12.2022
"""

import numpy as np
from matplotlib import pyplot as plt
import csv
from re import sub
import math


#########################################################################
##### Einstellungen für Plots ###########################################
#########################################################################

plt.rcParams['figure.dpi']  = 300
plt.rcParams['savefig.dpi'] = 300

#########################################################################
##### Definition der Knoten #############################################
#########################################################################

class class_Knoten:
    def __init__(self):
        self.p                 = 0        
        self.V                 = 0
        self.m                 = 0
        self.Q                 = 0  
        self.m_p               = 0
    def Reset_Knotenstroeme(self):
        self.Q                 = 0
        self.m_p               = 0
    def Reset_Knoten(self):
        self.p                 = 0        
        self.V                 = 0        
        self.m                 = 0
        self.Q                 = 0
        self.m_p               = 0        
       
class class_Datenlogger_Knoten:
    def __init__(self):
        self.p                 = []        
        self.Q                 = []        
        self.m_p               = []        
    def Werte_loeschen(self):
        self.p.clear()        
        self.Q.clear()        
        self.m_p.clear()        
    def Werte_anhaengen(self, Knoten):
        self.p.append(Knoten.p)        
        self.Q.append(Knoten.Q)        
        self.m_p.append(Knoten.m_p)        

class class_Datenlogger_Element:
    def __init__(self):
        self.Q         = []        
        self.m_p       = []        
    def Werte_loeschen(self):
        self.Q.clear()        
        self.m_p.clear()        
    def Werte_anhaengen(self, Element):
        self.Q.append(Element.Info_Q)        
        self.m_p.append(Element.Info_m_p)        

    
#########################################################################
##### DGL-Solver ########################################################
#########################################################################

class ODESolver:
    """ODESolver
    Solves ODE on the form:
    u' = f(u, t), u(0) = U0
    Parameters
    ----------
    f : callable
        Right-hand-side function f(u, t)
    """

    def __init__(self, f, KNOTEN, Klasse):
        if (Klasse == "class_DGL_Kapazitaet_Volumenstrombasiert"):
            self.Knoteni = KNOTEN[0]            
            self.STROEME = [0]                
        if (Klasse == "class_DGL_Kapazitaet_Massenstrombasiert"):
            self.Knoteni = KNOTEN[0]            
            self.STROEME = [0]                

        self.f = f
        self.number_of_eqns = 2
        U0in = [0,0]
        U0in = np.asarray(U0in)
        self.number_of_eqns = U0in.size                        
        self.U0 = U0in                     
        
    def aktualisiere_initial_conditions(self, Klasse):                
        if (Klasse == "class_DGL_Kapazitaet_Volumenstrombasiert"):
            self.STROEME = [self.Knoteni.Q]
            self.U0 = [self.Knoteni.p,0]                            
        if (Klasse == "class_DGL_Kapazitaet_Massenstrombasiert"):
            self.STROEME = [self.Knoteni.m_p]
            self.U0 = [self.Knoteni.p,0]                            
            
    def solve_timestep(self, delta_t):
        """
        Solves ODE according to given time points.
        The resolution is implied by spacing of 
        time points.
        Parameters
        ----------
        time_points : array_like
            Time points to solve for
        
        Returns
        -------
        u : array_like
            Solution
        t : array_like
            Time points corresponding to solution
        """

        time_points = [0, delta_t]
        self.t = np.asarray(time_points)
        n = self.t.size

        self.u = np.zeros((n, self.number_of_eqns))        
        
        self.u[0, :] = self.U0                
        # Integrate
        for i in range(n - 1):
            self.i = i
            self.u[i + 1] = self.advance()

        return self.u 

        def advance(self):
            """Advance solution one time step."""
            raise NotImplementedError
            
class RungeKutta4(ODESolver):
    def advance(self):
        u, f, i, t, STROEME = self.u, self.f, self.i, self.t, self.STROEME
        dt = t[i + 1] - t[i]
        dt2 = dt / 2
        K1 = dt * f(u[i, :], STROEME, t[i])
        K2 = dt * f(u[i, :] + 0.5 * K1, STROEME, t[i] + dt2)
        K3 = dt * f(u[i, :] + 0.5 * K2, STROEME, t[i] + dt2)
        K4 = dt * f(u[i, :] + K3, STROEME, t[i] + dt)
        return u[i, :] + (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)   
        
class Euler(ODESolver):
    def advance(self):
        u, f, i, t, STROEME = self.u, self.f, self.i, self.t, self.STROEME
        dt = t[i + 1] - t[i]
        dt2 = dt / 2
        K1 = dt * f(u[i, :], STROEME, t[i])        
        return u[i, :] + K1   
 


#########################################################################
##### Definition der sekundären Komponenten #############################
#########################################################################

class class_Knotenpunkt:
    def __init__(self, KNOTEN):
        self.Knoteni = KNOTEN[0]                
        
class class_Blende:
    def __init__(self, PARAMETER_FLUID, alpha, d, KNOTEN):
        self.Knoteni  = KNOTEN[0]
        self.Knotenj  = KNOTEN[1]        
        self.alpha    = alpha 
        self.d        = d        
        self.rho      = PARAMETER_FLUID[0]
        self.E        = PARAMETER_FLUID[1]    
        self.Info_Q   = 0
        self.Info_m_p = 0
        
    def Berechnung_Volumenstroeme(self):                        
        dp           = self.Knoteni.p - self.Knotenj.p
        A            = 3.14156 * pow(self.d,2)/4
        #Wenn dp zu klein, dann keine Berechnung und Rückgabe Q = 0
        if (dp > 1e-3):
            self.Knoteni.Q   -= self.alpha * A * pow((2/self.rho),0.5) * pow(dp,0.5)
            self.Knotenj.Q   += self.alpha * A * pow((2/self.rho),0.5) * pow(dp,0.5)        
            self.Info_Q       = self.Knotenj.Q
            
    def Berechnung_Massenstroeme(self):                        
        dp           = self.Knoteni.p - self.Knotenj.p
        A            = 3.14156 * pow(self.d,2)/4
        #Wenn dp zu klein, dann keine Berechnung und Rückgabe Q = 0
        if (dp > 1e-3):
            self.Knoteni.m_p   -= self.rho * self.alpha * A * pow((2/self.rho),0.5) * pow(dp,0.5)
            self.Knotenj.m_p   += self.rho * self.alpha * A * pow((2/self.rho),0.5) * pow(dp,0.5)        
            self.Info_m_p       = self.Knotenj.m_p
            
    
#########################################################################
##### Definition der primären Komponenten ###############################
#########################################################################
        
class class_DGL_Kapazitaet_Volumenstrombasiert:
    """
    Gleichungssystem
    du0/dt = u1
    du1/dt = g + Fi/m - Fj/m
    """
    """    
    """
    def __init__(self, PARAMETER_FLUID, KNOTEN):
        self.Knoteni = KNOTEN[0]                         
        self.rho     = PARAMETER_FLUID[0]
        self.E       = PARAMETER_FLUID[1]        
        self.Euler = Euler(self, KNOTEN, "class_DGL_Kapazitaet_Volumenstrombasiert")        
                       
    def Aktualisiere_Anfangsrandbedingungen(self):
        self.Euler.aktualisiere_initial_conditions("class_DGL_Kapazitaet_Volumenstrombasiert")    
        
    def Loesung_Differentialgleichung_Zeitschritt(self, delta_t):
        self.Aktualisiere_Anfangsrandbedingungen()
        Rueckgabevektor            =  self.Euler.solve_timestep(delta_t)
        self.Zuweisung_motion(Rueckgabevektor, delta_t)  
        self.Aktualisiere_Anfangsrandbedingungen()           

    def __call__(self, u, STROEME, t):
        u0, u1  = u
        V       = self.Knoteni.V        
        C       = V / self.E
        return np.array([(STROEME[0]/C),0])    

    def Zuweisung_motion(self, u, delta_t):        
        p_p  = u[1,1]        
        p    = u[1,0]                
        self.Knoteni.p      = p

class class_DGL_Kapazitaet_Massenstrombasiert:
    """
    Gleichungssystem
    du0/dt = u1
    du1/dt = g + Fi/m - Fj/m
    """
    """    
    """
    def __init__(self, PARAMETER_FLUID, KNOTEN):
        self.Knoteni = KNOTEN[0]                         
        self.rho     = PARAMETER_FLUID[0]
        self.E       = PARAMETER_FLUID[1]        
        self.Euler = Euler(self, KNOTEN, "class_DGL_Kapazitaet_Massenstrombasiert")        
                       
    def Aktualisiere_Anfangsrandbedingungen(self):
        self.Euler.aktualisiere_initial_conditions("class_DGL_Kapazitaet_Massenstrombasiert")    
        
    def Loesung_Differentialgleichung_Zeitschritt(self, delta_t):
        self.Aktualisiere_Anfangsrandbedingungen()
        Rueckgabevektor            =  self.Euler.solve_timestep(delta_t)
        self.Zuweisung_motion(Rueckgabevektor, delta_t)  
        self.Aktualisiere_Anfangsrandbedingungen()           

    def __call__(self, u, STROEME, t):
        u0, u1  = u
        E       = self.E
        rho     = self.rho
        V       = self.Knoteni.V
        m       = rho * V
        V_p     = 0
        
        return np.array([(E*((STROEME[0]/m)-(V_p/V))),0])    

    def Zuweisung_motion(self, u, delta_t):        
        p_p  = u[1,1]        
        p    = u[1,0]                
        self.Knoteni.p      = p
        
#########################################################################
##### Erstellung aller Knoten ###########################################
#########################################################################

K1            = class_Knoten()
K2            = class_Knoten()
K3            = class_Knoten()
K4            = class_Knoten()
K5            = class_Knoten()
K6            = class_Knoten()
K7            = class_Knoten()
K8            = class_Knoten()
K9            = class_Knoten()
K10           = class_Knoten()
K11           = class_Knoten()
K12           = class_Knoten()
K13           = class_Knoten()
K14           = class_Knoten()
K15           = class_Knoten()
K16           = class_Knoten()
K17           = class_Knoten()
K18           = class_Knoten()
K19           = class_Knoten()
K20           = class_Knoten()

#########################################################################
##### Reset_Knotenkraft #################################################
#########################################################################
        
    
def Reset_Knotenstroeme():
    K1.Reset_Knotenstroeme()
    K2.Reset_Knotenstroeme()
    K3.Reset_Knotenstroeme()
    K4.Reset_Knotenstroeme()
    K5.Reset_Knotenstroeme()
    K6.Reset_Knotenstroeme()
    K7.Reset_Knotenstroeme()
    K8.Reset_Knotenstroeme()
    K9.Reset_Knotenstroeme()
    K10.Reset_Knotenstroeme()
    K11.Reset_Knotenstroeme()
    K12.Reset_Knotenstroeme()
    K13.Reset_Knotenstroeme()
    K14.Reset_Knotenstroeme()
    K15.Reset_Knotenstroeme()
    K16.Reset_Knotenstroeme()
    K17.Reset_Knotenstroeme()
    K18.Reset_Knotenstroeme()
    K19.Reset_Knotenstroeme()
    K20.Reset_Knotenstroeme()


def Reset_Knoten():
    K1.Reset_Knoten()
    K2.Reset_Knoten()
    K3.Reset_Knoten()
    K4.Reset_Knoten()
    K5.Reset_Knoten()
    K6.Reset_Knoten()
    K7.Reset_Knoten()
    K8.Reset_Knoten()
    K9.Reset_Knoten()
    K10.Reset_Knoten()
    K11.Reset_Knoten()
    K12.Reset_Knoten()
    K13.Reset_Knoten()
    K14.Reset_Knoten()
    K15.Reset_Knoten()
    K16.Reset_Knoten()
    K17.Reset_Knoten()
    K18.Reset_Knoten()
    K19.Reset_Knoten()
    K20.Reset_Knoten()

#########################################################################
##### Datenlogger #######################################################
#########################################################################

datenlogger_t                  = []

datenlogger_K1        = class_Datenlogger_Knoten()
datenlogger_K2        = class_Datenlogger_Knoten()
datenlogger_K3        = class_Datenlogger_Knoten()
datenlogger_K4        = class_Datenlogger_Knoten()
datenlogger_K5        = class_Datenlogger_Knoten()
datenlogger_K6        = class_Datenlogger_Knoten()
datenlogger_K7        = class_Datenlogger_Knoten()
datenlogger_K8        = class_Datenlogger_Knoten()
datenlogger_K9        = class_Datenlogger_Knoten()
datenlogger_K10       = class_Datenlogger_Knoten()
datenlogger_K11       = class_Datenlogger_Knoten()
datenlogger_K12       = class_Datenlogger_Knoten()
datenlogger_K13       = class_Datenlogger_Knoten()
datenlogger_K14       = class_Datenlogger_Knoten()
datenlogger_K15       = class_Datenlogger_Knoten()
datenlogger_K16       = class_Datenlogger_Knoten()
datenlogger_K17       = class_Datenlogger_Knoten()
datenlogger_K18       = class_Datenlogger_Knoten()
datenlogger_K19       = class_Datenlogger_Knoten()
datenlogger_K20       = class_Datenlogger_Knoten()

datenlogger_E1        = class_Datenlogger_Element()
datenlogger_E2        = class_Datenlogger_Element()
datenlogger_E3        = class_Datenlogger_Element()
datenlogger_E4        = class_Datenlogger_Element()
datenlogger_E5        = class_Datenlogger_Element()
datenlogger_E6        = class_Datenlogger_Element()
datenlogger_E7        = class_Datenlogger_Element()
datenlogger_E8        = class_Datenlogger_Element()
datenlogger_E9        = class_Datenlogger_Element()
datenlogger_E10       = class_Datenlogger_Element()
datenlogger_E11       = class_Datenlogger_Element()
datenlogger_E12       = class_Datenlogger_Element()
datenlogger_E13       = class_Datenlogger_Element()
datenlogger_E14       = class_Datenlogger_Element()
datenlogger_E15       = class_Datenlogger_Element()
datenlogger_E16       = class_Datenlogger_Element()
datenlogger_E17       = class_Datenlogger_Element()
datenlogger_E18       = class_Datenlogger_Element()
datenlogger_E19       = class_Datenlogger_Element()
datenlogger_E20       = class_Datenlogger_Element()
    
def Datenlogger_leeren():
    datenlogger_t.clear()

    datenlogger_K1.Werte_loeschen()
    datenlogger_K2.Werte_loeschen()
    datenlogger_K3.Werte_loeschen()
    datenlogger_K4.Werte_loeschen()
    datenlogger_K5.Werte_loeschen()
    datenlogger_K6.Werte_loeschen()
    datenlogger_K7.Werte_loeschen()
    datenlogger_K8.Werte_loeschen()
    datenlogger_K9.Werte_loeschen()
    datenlogger_K10.Werte_loeschen()
    datenlogger_K11.Werte_loeschen()
    datenlogger_K12.Werte_loeschen()
    datenlogger_K13.Werte_loeschen()
    datenlogger_K14.Werte_loeschen()
    datenlogger_K15.Werte_loeschen()
    datenlogger_K16.Werte_loeschen()
    datenlogger_K17.Werte_loeschen()
    datenlogger_K18.Werte_loeschen()
    datenlogger_K19.Werte_loeschen()
    datenlogger_K20.Werte_loeschen()

    datenlogger_E1.Werte_loeschen()
    datenlogger_E2.Werte_loeschen()
    datenlogger_E3.Werte_loeschen()
    datenlogger_E4.Werte_loeschen()
    datenlogger_E5.Werte_loeschen()
    datenlogger_E6.Werte_loeschen()
    datenlogger_E7.Werte_loeschen()
    datenlogger_E8.Werte_loeschen()
    datenlogger_E9.Werte_loeschen()
    datenlogger_E10.Werte_loeschen()
    datenlogger_E11.Werte_loeschen()
    datenlogger_E12.Werte_loeschen()
    datenlogger_E13.Werte_loeschen()
    datenlogger_E14.Werte_loeschen()
    datenlogger_E15.Werte_loeschen()
    datenlogger_E16.Werte_loeschen()
    datenlogger_E17.Werte_loeschen()
    datenlogger_E18.Werte_loeschen()
    datenlogger_E19.Werte_loeschen()
    datenlogger_E20.Werte_loeschen()

def Datenlogger_Knoten_schreiben(t):
    datenlogger_t.append(t) 
    datenlogger_K1.Werte_anhaengen(K1)           
    datenlogger_K2.Werte_anhaengen(K2)           
    datenlogger_K3.Werte_anhaengen(K3)           
    datenlogger_K4.Werte_anhaengen(K4)           
    datenlogger_K5.Werte_anhaengen(K5)           
    datenlogger_K6.Werte_anhaengen(K6)           
    datenlogger_K7.Werte_anhaengen(K7)           
    datenlogger_K8.Werte_anhaengen(K8)           
    datenlogger_K9.Werte_anhaengen(K9)           
    datenlogger_K10.Werte_anhaengen(K10)           
    datenlogger_K11.Werte_anhaengen(K11)           
    datenlogger_K12.Werte_anhaengen(K12)           
    datenlogger_K13.Werte_anhaengen(K13)           
    datenlogger_K14.Werte_anhaengen(K14)           
    datenlogger_K15.Werte_anhaengen(K15)           
    datenlogger_K16.Werte_anhaengen(K16)           
    datenlogger_K17.Werte_anhaengen(K17)           
    datenlogger_K18.Werte_anhaengen(K18)           
    datenlogger_K19.Werte_anhaengen(K19)           
    datenlogger_K20.Werte_anhaengen(K20)     
    
    
#########################################################################
##### csv-Datei schreiben ###############################################
#########################################################################

def comma_float(num):
            ''' wandelt float in string mit echtem Komma um '''
            return sub(r'\.', ',', str(num))
    
def csv_schreiben(Dateiname, Knoten):
    with open(Dateiname, 'w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=";")
        writer.writerow(["t","p"])
        for point in range(len(datenlogger_t)):            
            writer.writerow([comma_float(datenlogger_t[point]), comma_float(Knoten.p[point])])
            