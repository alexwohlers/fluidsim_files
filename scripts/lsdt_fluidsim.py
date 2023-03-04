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
##### Definition der Elemente E #########################################
#########################################################################

class class_Knotenpunkt:
    def __init__(self, KNOTEN):
        self.Knoteni = KNOTEN[0]                

"""
Bedeutungen:
    _R  : Resistiver Widerstand
    _L  : Induktiver Widerstand
    _RL : Resistiver und Induktiver Widerstand
"""

class class_Blende_R:
    """
    Berücksichtigung des resistiven Widerstandes einer Blende
    """
    def __init__(self, PARAMETER_FLUID, alpha, d, KNOTEN):
        self.Knoteni  = KNOTEN[0]
        self.Knotenj  = KNOTEN[1]        
        self.alpha    = alpha 
        self.d        = d        
        self.rho      = PARAMETER_FLUID[0]
        self.E        = PARAMETER_FLUID[1]    
        self.Info_Q   = 0
        self.Info_m_p = 0
        
    def Berechnung_Volumenstroeme(self, dt):                        
        dp           = self.Knoteni.p - self.Knotenj.p
        A            = 3.14156 * pow(self.d,2)/4
        #Wenn dp zu klein, dann keine Berechnung und Rückgabe Q = 0
        if (dp > 1e-3):
            self.Knoteni.Q   -= self.alpha * A * pow((2/self.rho),0.5) * pow(dp,0.5)
            self.Knotenj.Q   += self.alpha * A * pow((2/self.rho),0.5) * pow(dp,0.5)        
            self.Info_Q       = self.Knotenj.Q
            
    def Berechnung_Massenstroeme(self, dt):                        
        dp           = self.Knoteni.p - self.Knotenj.p
        A            = 3.14156 * pow(self.d,2)/4
        #Wenn dp zu klein, dann keine Berechnung und Rückgabe Q = 0
        if (dp > 1e-3):
            self.Knoteni.m_p   -= self.rho * self.alpha * A * pow((2/self.rho),0.5) * pow(dp,0.5)
            self.Knotenj.m_p   += self.rho * self.alpha * A * pow((2/self.rho),0.5) * pow(dp,0.5)        
            self.Info_m_p       = self.Knotenj.m_p
            

class class_Rohrleitung_L:
    """
    Berücksichtigung des induktiven Widerstandes einer Rohrleitung
    """
    def __init__(self, PARAMETER_FLUID, l, d, KNOTEN):
        self.Knoteni  = KNOTEN[0]
        self.Knotenj  = KNOTEN[1]        
        self.l        = l
        self.d        = d        
        self.rho      = PARAMETER_FLUID[0]
        self.E        = PARAMETER_FLUID[1]    
        self.Info_Q   = 0
        self.Info_m_p = 0
        
    def Berechnung_Volumenstroeme(self, dt):                        
        dp           = self.Knoteni.p - self.Knotenj.p
        A            = 3.14156 * pow(self.d,2)/4
        L            = self.rho * self.l / A
        #Wenn dp zu klein, dann keine Berechnung und Rückgabe Q = 0
        if (dp > 1e-3):
            self.Knoteni.Q   += 1 / L * dp * dt
            self.Knotenj.Q   -= 1 / L * dp * dt
            self.Info_Q       = self.Knotenj.Q
            
    def Berechnung_Massenstroeme(self, dt):                        
        dp           = self.Knoteni.p - self.Knotenj.p
        A            = 3.14156 * pow(self.d,2)/4
        L            = self.rho * self.l / A
        #Wenn dp zu klein, dann keine Berechnung und Rückgabe Q = 0
        if (dp > 1e-3):
            self.Knoteni.m_p  += self.rho / L * dp * dt
            self.Knotenj.m_p  -= self.rho / L * dp * dt
            self.Info_m_p      = self.Knotenj.Q
            
#########################################################################
##### Definition der Knotenelemente EK ##################################
#########################################################################
        
class class_DGL_Behaelter_starr_Volumenstrombasiert:
    """
    Starrer Behälter    
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

class class_DGL_Behaelter_starr_Massenstrombasiert:
    """
    Starrer Behälter   
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
K21           = class_Knoten()
K22           = class_Knoten()
K23           = class_Knoten()
K24           = class_Knoten()
K25           = class_Knoten()
K26           = class_Knoten()
K27           = class_Knoten()
K28           = class_Knoten()
K29           = class_Knoten()
K30           = class_Knoten()
K31           = class_Knoten()
K32           = class_Knoten()
K33           = class_Knoten()
K34           = class_Knoten()
K35           = class_Knoten()
K36           = class_Knoten()
K37           = class_Knoten()
K38           = class_Knoten()
K39           = class_Knoten()
K40           = class_Knoten()
K41           = class_Knoten()
K42           = class_Knoten()
K43           = class_Knoten()
K44           = class_Knoten()
K45           = class_Knoten()
K46           = class_Knoten()
K47           = class_Knoten()
K48           = class_Knoten()
K49           = class_Knoten()
K50           = class_Knoten()

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
    K21.Reset_Knotenstroeme()
    K22.Reset_Knotenstroeme()
    K23.Reset_Knotenstroeme()
    K24.Reset_Knotenstroeme()
    K25.Reset_Knotenstroeme()
    K26.Reset_Knotenstroeme()
    K27.Reset_Knotenstroeme()
    K28.Reset_Knotenstroeme()
    K29.Reset_Knotenstroeme()
    K30.Reset_Knotenstroeme()
    K31.Reset_Knotenstroeme()
    K32.Reset_Knotenstroeme()
    K33.Reset_Knotenstroeme()
    K34.Reset_Knotenstroeme()
    K35.Reset_Knotenstroeme()
    K36.Reset_Knotenstroeme()
    K37.Reset_Knotenstroeme()
    K38.Reset_Knotenstroeme()
    K39.Reset_Knotenstroeme()
    K40.Reset_Knotenstroeme()
    K41.Reset_Knotenstroeme()
    K42.Reset_Knotenstroeme()
    K43.Reset_Knotenstroeme()
    K44.Reset_Knotenstroeme()
    K45.Reset_Knotenstroeme()
    K46.Reset_Knotenstroeme()
    K47.Reset_Knotenstroeme()
    K48.Reset_Knotenstroeme()
    K49.Reset_Knotenstroeme()
    K50.Reset_Knotenstroeme()


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
    K21.Reset_Knoten()
    K22.Reset_Knoten()
    K23.Reset_Knoten()
    K24.Reset_Knoten()
    K25.Reset_Knoten()
    K26.Reset_Knoten()
    K27.Reset_Knoten()
    K28.Reset_Knoten()
    K29.Reset_Knoten()
    K30.Reset_Knoten()
    K31.Reset_Knoten()
    K32.Reset_Knoten()
    K33.Reset_Knoten()
    K34.Reset_Knoten()
    K35.Reset_Knoten()
    K36.Reset_Knoten()
    K37.Reset_Knoten()
    K38.Reset_Knoten()
    K39.Reset_Knoten()
    K40.Reset_Knoten()
    K41.Reset_Knoten()
    K42.Reset_Knoten()
    K43.Reset_Knoten()
    K44.Reset_Knoten()
    K45.Reset_Knoten()
    K46.Reset_Knoten()
    K47.Reset_Knoten()
    K48.Reset_Knoten()
    K49.Reset_Knoten()
    K50.Reset_Knoten()

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
datenlogger_K21       = class_Datenlogger_Knoten()
datenlogger_K22       = class_Datenlogger_Knoten()
datenlogger_K23       = class_Datenlogger_Knoten()
datenlogger_K24       = class_Datenlogger_Knoten()
datenlogger_K25       = class_Datenlogger_Knoten()
datenlogger_K26       = class_Datenlogger_Knoten()
datenlogger_K27       = class_Datenlogger_Knoten()
datenlogger_K28       = class_Datenlogger_Knoten()
datenlogger_K29       = class_Datenlogger_Knoten()
datenlogger_K30       = class_Datenlogger_Knoten()
datenlogger_K31       = class_Datenlogger_Knoten()
datenlogger_K32       = class_Datenlogger_Knoten()
datenlogger_K33       = class_Datenlogger_Knoten()
datenlogger_K34       = class_Datenlogger_Knoten()
datenlogger_K35       = class_Datenlogger_Knoten()
datenlogger_K36       = class_Datenlogger_Knoten()
datenlogger_K37       = class_Datenlogger_Knoten()
datenlogger_K38       = class_Datenlogger_Knoten()
datenlogger_K39       = class_Datenlogger_Knoten()
datenlogger_K40       = class_Datenlogger_Knoten()
datenlogger_K41       = class_Datenlogger_Knoten()
datenlogger_K42       = class_Datenlogger_Knoten()
datenlogger_K43       = class_Datenlogger_Knoten()
datenlogger_K44       = class_Datenlogger_Knoten()
datenlogger_K45       = class_Datenlogger_Knoten()
datenlogger_K46       = class_Datenlogger_Knoten()
datenlogger_K47       = class_Datenlogger_Knoten()
datenlogger_K48       = class_Datenlogger_Knoten()
datenlogger_K49       = class_Datenlogger_Knoten()
datenlogger_K50       = class_Datenlogger_Knoten()

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
datenlogger_E21       = class_Datenlogger_Element()
datenlogger_E22       = class_Datenlogger_Element()
datenlogger_E23       = class_Datenlogger_Element()
datenlogger_E24       = class_Datenlogger_Element()
datenlogger_E25       = class_Datenlogger_Element()
datenlogger_E26       = class_Datenlogger_Element()
datenlogger_E27       = class_Datenlogger_Element()
datenlogger_E28       = class_Datenlogger_Element()
datenlogger_E29       = class_Datenlogger_Element()
datenlogger_E30       = class_Datenlogger_Element()
datenlogger_E31       = class_Datenlogger_Element()
datenlogger_E32       = class_Datenlogger_Element()
datenlogger_E33       = class_Datenlogger_Element()
datenlogger_E34       = class_Datenlogger_Element()
datenlogger_E35       = class_Datenlogger_Element()
datenlogger_E36       = class_Datenlogger_Element()
datenlogger_E37       = class_Datenlogger_Element()
datenlogger_E38       = class_Datenlogger_Element()
datenlogger_E39       = class_Datenlogger_Element()
datenlogger_E40       = class_Datenlogger_Element()
datenlogger_E41       = class_Datenlogger_Element()
datenlogger_E42       = class_Datenlogger_Element()
datenlogger_E43       = class_Datenlogger_Element()
datenlogger_E44       = class_Datenlogger_Element()
datenlogger_E45       = class_Datenlogger_Element()
datenlogger_E46       = class_Datenlogger_Element()
datenlogger_E47       = class_Datenlogger_Element()
datenlogger_E48       = class_Datenlogger_Element()
datenlogger_E49       = class_Datenlogger_Element()
datenlogger_E50       = class_Datenlogger_Element()

    
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
    datenlogger_K21.Werte_loeschen()
    datenlogger_K22.Werte_loeschen()
    datenlogger_K23.Werte_loeschen()
    datenlogger_K24.Werte_loeschen()
    datenlogger_K25.Werte_loeschen()
    datenlogger_K26.Werte_loeschen()
    datenlogger_K27.Werte_loeschen()
    datenlogger_K28.Werte_loeschen()
    datenlogger_K29.Werte_loeschen()
    datenlogger_K30.Werte_loeschen()
    datenlogger_K31.Werte_loeschen()
    datenlogger_K32.Werte_loeschen()
    datenlogger_K33.Werte_loeschen()
    datenlogger_K34.Werte_loeschen()
    datenlogger_K35.Werte_loeschen()
    datenlogger_K36.Werte_loeschen()
    datenlogger_K37.Werte_loeschen()
    datenlogger_K38.Werte_loeschen()
    datenlogger_K39.Werte_loeschen()
    datenlogger_K40.Werte_loeschen()
    datenlogger_K41.Werte_loeschen()
    datenlogger_K42.Werte_loeschen()
    datenlogger_K43.Werte_loeschen()
    datenlogger_K44.Werte_loeschen()
    datenlogger_K45.Werte_loeschen()
    datenlogger_K46.Werte_loeschen()
    datenlogger_K47.Werte_loeschen()
    datenlogger_K48.Werte_loeschen()
    datenlogger_K49.Werte_loeschen()
    datenlogger_K50.Werte_loeschen()

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
    datenlogger_E21.Werte_loeschen()
    datenlogger_E22.Werte_loeschen()
    datenlogger_E23.Werte_loeschen()
    datenlogger_E24.Werte_loeschen()
    datenlogger_E25.Werte_loeschen()
    datenlogger_E26.Werte_loeschen()
    datenlogger_E27.Werte_loeschen()
    datenlogger_E28.Werte_loeschen()
    datenlogger_E29.Werte_loeschen()
    datenlogger_E30.Werte_loeschen()
    datenlogger_E31.Werte_loeschen()
    datenlogger_E32.Werte_loeschen()
    datenlogger_E33.Werte_loeschen()
    datenlogger_E34.Werte_loeschen()
    datenlogger_E35.Werte_loeschen()
    datenlogger_E36.Werte_loeschen()
    datenlogger_E37.Werte_loeschen()
    datenlogger_E38.Werte_loeschen()
    datenlogger_E39.Werte_loeschen()
    datenlogger_E40.Werte_loeschen()
    datenlogger_E41.Werte_loeschen()
    datenlogger_E42.Werte_loeschen()
    datenlogger_E43.Werte_loeschen()
    datenlogger_E44.Werte_loeschen()
    datenlogger_E45.Werte_loeschen()
    datenlogger_E46.Werte_loeschen()
    datenlogger_E47.Werte_loeschen()
    datenlogger_E48.Werte_loeschen()
    datenlogger_E49.Werte_loeschen()
    datenlogger_E50.Werte_loeschen()

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
    datenlogger_K21.Werte_anhaengen(K21)           
    datenlogger_K22.Werte_anhaengen(K22)           
    datenlogger_K23.Werte_anhaengen(K23)           
    datenlogger_K24.Werte_anhaengen(K24)           
    datenlogger_K25.Werte_anhaengen(K25)           
    datenlogger_K26.Werte_anhaengen(K26)           
    datenlogger_K27.Werte_anhaengen(K27)           
    datenlogger_K28.Werte_anhaengen(K28)           
    datenlogger_K29.Werte_anhaengen(K29)           
    datenlogger_K30.Werte_anhaengen(K30)  
    datenlogger_K31.Werte_anhaengen(K31)           
    datenlogger_K32.Werte_anhaengen(K32)           
    datenlogger_K33.Werte_anhaengen(K33)           
    datenlogger_K34.Werte_anhaengen(K34)           
    datenlogger_K35.Werte_anhaengen(K35)           
    datenlogger_K36.Werte_anhaengen(K36)           
    datenlogger_K37.Werte_anhaengen(K37)           
    datenlogger_K38.Werte_anhaengen(K38)           
    datenlogger_K39.Werte_anhaengen(K39)           
    datenlogger_K40.Werte_anhaengen(K40) 
    datenlogger_K41.Werte_anhaengen(K41)           
    datenlogger_K42.Werte_anhaengen(K42)           
    datenlogger_K43.Werte_anhaengen(K43)           
    datenlogger_K44.Werte_anhaengen(K44)           
    datenlogger_K45.Werte_anhaengen(K45)           
    datenlogger_K46.Werte_anhaengen(K46)           
    datenlogger_K47.Werte_anhaengen(K47)           
    datenlogger_K48.Werte_anhaengen(K48)           
    datenlogger_K49.Werte_anhaengen(K49)           
    datenlogger_K50.Werte_anhaengen(K50)     
    
    
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
            