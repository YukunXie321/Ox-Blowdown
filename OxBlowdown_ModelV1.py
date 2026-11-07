import sys
import CoolProp
from CoolProp.CoolProp import PropsSI
from dataclasses import dataclass
from scipy.integrate import solve_ivp #RK45 function import
import matplotlib.pyplot as plt #graphing library

@dataclass
class Inputs:

    #Initial Volume (m^3)
    V: float = 0.011

    #Initial temperature (K)
    T_initial: float = 295

    #Initial Ullage (V_v/V_l)
    f: float = 0.2

    #Initial Mass Flow rate (kg/s)
    mdot: float = 1


@dataclass

class InitialStates:

    #Initial Pressure (Pa)
    P_initial: float 

    #Liquid Density (kg/m^3)
    density_liquid: float 

    #Vapour Density (kg/m^3)
    density_vapour: float 

    #Mass of vapour (kg) (Vapour volume*vapour density)
    M_v: float 

    #Mass of liquid
    M_l: float

    #Total Mass (kg)
    M_total: float 

    #Total Internal Energy
    U_total: float 

    V_vapour: float
    V_liquid: float
    U_liquid: float
    U_vapour: float



#Calculate everything that depends on the initial state
def initial_state(var: Inputs):
    P_initial: float = PropsSI("P", "T", var.T_initial, "Q", 0, "NitrousOxide")
    density_liquid: float = PropsSI("Dmass", "T", var.T_initial, "Q", 0, "NitrousOxide")
    density_vapour: float = PropsSI("Dmass", "T", var.T_initial, "Q", 1, "NitrousOxide")
    V_liquid: float = var.V / (1 + var.f)
    V_vapour: float = var.f * var.V / (1 + var.f)
    U_liquid: float = PropsSI("Umass", "T", var.T_initial, "Q", 0, "NitrousOxide")
    U_vapour: float = PropsSI("Umass", "T", var.T_initial, "Q", 1, "NitrousOxide")
    M_v: float = density_vapour * V_vapour
    M_l: float = density_liquid * V_liquid
    M_total: float = M_l + M_v
    U_total: float = M_l * U_liquid + M_v * U_vapour

    return InitialStates(
        P_initial=P_initial,
        density_liquid=density_liquid,
        density_vapour=density_vapour,
        U_liquid=U_liquid,
        U_vapour=U_vapour,
        V_liquid=V_liquid,
        V_vapour=V_vapour,
        M_v=M_v,
        M_l=M_l,
        M_total=M_total,
        U_total=U_total)


@dataclass
class Parameters:
    T: float        
    P: float        
    x: float        

#Current conditions of the tank

def current_State(M_total: float, U_total: float, var: Inputs) -> Parameters:
    new_density: float = M_total / var.V
    new_u: float = U_total / M_total

    T: float = PropsSI("T", "Dmass", new_density, "Umass", new_u, "NitrousOxide")
    P: float = PropsSI("P", "Dmass", new_density, "Umass", new_u, "NitrousOxide")
    x: float = PropsSI("Q", "Dmass", new_density, "Umass", new_u, "NitrousOxide")


    return Parameters(T=T, P=P, x=x)

#Using the current state to determine the new rate of change
def rate(t: float, y: list, var: Inputs) -> list:
    M_total, U_total = y 
    s = current_State(M_total, U_total, var)
    h_l: float = PropsSI("H", "T", s.T, "Q", 0, "NitrousOxide")
    dM_dt: float = -var.mdot
    dU_dt: float = -var.mdot * h_l
    return [dM_dt, dU_dt]

#Stop conditions

def liquid_gone(t: float, y: list, var: Inputs) -> float:
    M_total, U_total = y
    s = current_State(M_total, U_total, var)
    return 1.0 - s.x       

liquid_gone.terminal = True          
liquid_gone.direction = -1    

#RK45 solve
def RK45(var: Inputs):
    start = initial_state(var)
    y = [start.M_total, start.U_total]

    t_end = start.M_total / var.mdot #time to remove all the mass

    #RK45 specific function from ClaudeAI
    sol = solve_ivp(
        rate,                                        # your rate function (the physics)
        (0, t_end),                                  # time span: from 0 s to t_end
        y,                                          # starting values of y
        method="RK45",
        args=(var,),                                 # passed to rate() and liquid_gone() as var
        events=liquid_gone,                          # stop when the liquid is gone
        rtol=1e-6,                                   # relative error tolerance
        atol=[1e-6, 1.0],                            # absolute tolerance: [kg, J]
        max_step=0.1,                                # largest time step allowed (s)
    )

    return sol


def store_data (sol, var: Inputs):
    t = list(sol.t)
    M_total = list(sol.y[0])
    U_total = list(sol.y[1])

    T = []
    P = []
    x = []
    i = 0
    while i < len(t):
        T.append(current_State(M_total[i], U_total[i], var).T); 
        P.append(current_State(M_total[i], U_total[i], var).P); 
        x.append(current_State(M_total[i], U_total[i], var).x)
        i += 1 

    return t, M_total, U_total, T, P, x

def plot_results(t: list, T: list, P: list):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6)) #Plot size and axis, change if different configuration is needed
    axes[0].plot(t, T)
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("N2O tank blowdown")
    axes[1].plot(t, P)
    axes[1].set_ylabel("Pressure (Pa)")
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()



#Testing

testInput = Inputs()
test1 = initial_state(testInput)
sol = RK45(testInput)
t, M_total, U_total, T, P, x = store_data(sol, testInput)
print(testInput)
plot_results(t, T, P)

