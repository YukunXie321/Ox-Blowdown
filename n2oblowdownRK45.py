"""
Simulation of Temperature and Pressure in tank during blowdown under 
a constant mass flow rate. Liquid vapour distribution is also tracked and 
plotted.
Mass and energy are integrated using the RK45 method. CoolProp
is used to querry thermodynamic properties.

ASSUMPTIONS
-Tank is rigid, so constant volume
-Liquid n2o leaves outlet, none enters
-No shaft/boundary work is performed
-Adiabatic tank, no heat transfer through walls
-Kinetic and potential energy changes are negligible
-Liquid and vapour are always in thermodynamic equilibrium
-Liquid N2o exits at constant mass flow rate
-Pure N2o with no pressurant gas
"""

from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# DEFINING FUNCTIONS

# Mass and energy differential equations
def derivatives(t, y):

    m, U = y

    #Temperature from mixture density(m/v) and specic internal energy(U/m) 
    T = PropsSI("T", "Dmass", m/V, "Umass", U/m, fluid)
    #Saturated liquid enthalpy at that temp
    h_l = PropsSI("Hmass", "T", T, "Q", 0, fluid)

    dmdt = -mdot
    dUdt = -mdot * h_l

    return [dmdt, dUdt]


# Vapour quality from tank state
def quality(t, y):

    m, U = y

    T = PropsSI("T", "Dmass", m/V, "Umass", U/m, fluid)
    #Saturated densities 
    rho_l = PropsSI("Dmass", "T", T, "Q", 0, fluid)
    rho_v = PropsSI("Dmass", "T", T, "Q", 1, fluid)
    
    #Vapour mass fraction calc
    x = (V/m - 1/rho_l) / (1/rho_v - 1/rho_l)

    return x


# stopping conditions for integration
#Condition 1, liquid depleted
def liquid_empty(t, y):
    return 1 - quality(t, y)

liquid_empty.terminal = True
liquid_empty.direction = -1


#Condition 2, vapor depleted
def vapour_empty(t, y):
    return quality(t, y)

vapour_empty.terminal = True
vapour_empty.direction = -1

# 1. USER INPUTS

fluid = "NitrousOxide"
V = 10 / 1000       # Tank volume (m^3)
P0_psi = 725.189   # Initial pressure (psi absolute)
ullage = 0.20       # Initial vapour volume fraction
mdot = 0.06         # Liquid mass flow rate (kg/s)

duration = 160      # Maximum simulation time (s)

# Convert pressure to Pa for CoolProp
Pa_per_psi = 6894.757293
P0 = P0_psi * Pa_per_psi



# 2. INITIAL TANK CONDITIONS


# Saturated liquid and vapour properties at P0
#Density
rho_l = PropsSI("Dmass", "P", P0, "Q", 0, fluid)
rho_v = PropsSI("Dmass", "P", P0, "Q", 1, fluid)

#specific internal energy of liquid and vapor
u_l = PropsSI("Umass", "P", P0, "Q", 0, fluid)
u_v = PropsSI("Umass", "P", P0, "Q", 1, fluid)

# Initial liquid and vapour masses
m_l = (1 - ullage) * V * rho_l
m_v = ullage * V * rho_v

# Initial total mass and internal energy
m = m_l + m_v
U = m_l * u_l + m_v * u_v


# 3. ARRAYS TO STORE RESULTS

times = []
pressures = []
temperatures = []
total_masses = []
liquid_masses = []
vapour_masses = []
ullages = []
qualities = []


# 4. SIMULATE BLOWDOWN USING RK45

# Solve the differential equations for m and U using RK45
solution = solve_ivp(
    derivatives,
    [0, duration],
    [m, U],
    method="RK45",
    events=[liquid_empty, vapour_empty],
    max_step=0.1,
    #relative tolerance
    rtol=1e-7,
    #absolute tolerance
    atol=[1e-9, 1e-5]
)

if not solution.success:
    raise RuntimeError(solution.message)


# Calculate properties at each saved m and U 
for t, m, U in zip(solution.t, solution.y[0], solution.y[1]):

    #Temperature and pressure 
    T = PropsSI("T", "Dmass", m/V, "Umass", U/m, fluid)
    P = PropsSI("P", "T", T, "Q", 0, fluid)
    
    #Quality to determine liquid and vapour mass
    x = quality(t, [m, U])
    x = max(0, min(1, x))  # Avoid tiny rounding errors

    
    m_l = m * (1 - x)
    m_v = m * x
    
    #Ullage calc using vapour density and vapour mass
    rho_v = PropsSI("Dmass", "T", T, "Q", 1, fluid)
    ullageCurrent = (m_v / rho_v) / V

    # Store results
    times.append(t)
    pressures.append(P / Pa_per_psi)
    temperatures.append(T - 273.15)

    total_masses.append(m)
    liquid_masses.append(m_l)
    vapour_masses.append(m_v)

    ullages.append(ullageCurrent * 100)
    qualities.append(x * 100)




# 5. PRINT FINAL RESULTS


print(f"Final time: {times[-1]:.2f} s")
print(f"Final pressure: {pressures[-1]:.3f} psi absolute")
print(f"Final temperature: {temperatures[-1]:.5f} °C")
print(f"Final liquid mass: {liquid_masses[-1]:.5f} kg")
print(f"Final vapour mass: {vapour_masses[-1]:.5f} kg")

if len(solution.t_events[0]) > 0:
    print("Stopped: Liquid depleted")

elif len(solution.t_events[1]) > 0:
    print("Stopped: Vapour depleted")

else:
    print("Stopped: Maximum simulation time reached")

# 6. PLOT RESULTS


# Create four subplots
fig, axes = plt.subplots(
    2, 2, figsize=(11, 7.5),
    sharex=True, constrained_layout=True
)

# Colours
blue = "#206596"
orange = "#c25429"
green = "#208374"
purple = "#8662a8"
grey = "#424854"

# Plot 1: Tank pressure
ax = axes[0, 0]
ax.plot(times, pressures, color=blue, linewidth=2.3)
ax.set_ylabel("Pressure [psia]")
ax.set_title("Tank pressure")

# Plot 2: Temperature
ax = axes[0, 1]
ax.plot(times, temperatures, color=orange, linewidth=2.3,
        label="Liquid and vapour (same temperature)")
ax.set_ylabel("Temperature [°C]")
ax.set_title("Equilibrium temperature")
ax.legend(fontsize=9, frameon=False)

# Plot 3: Mass remaining
ax = axes[1, 0]
ax.plot(times, total_masses, color=grey, linestyle="--",
        label="Total")
ax.plot(times, liquid_masses, color=blue, linewidth=2,
        label="Liquid")
ax.plot(times, vapour_masses, color=green, linewidth=2,
        label="Vapour")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Mass [kg]")
ax.set_title("Mass remaining")
ax.legend(frameon=False)

# Plot 4: Ullage and vapour quality
ax = axes[1, 1]
ax.plot(times, ullages, color=green, linewidth=2,
        label="Ullage (volume fraction)")
ax.plot(times, qualities, color=purple, linewidth=2,
        label="Quality (mass fraction)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Fraction [%]")
ax.set_ylim(0, 102)
ax.set_title("Vapour volume versus vapour mass")
ax.legend(frameon=False)

# Formatting for all four graphs
for ax in axes.flat:
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Main figure title
fig.suptitle(
    f"Two-phase N₂O blowdown | {V*1000:g} L, "
    f"{mdot:g} kg/s, adiabatic\n"
    f"Initial: {P0_psi:g} psi absolute, "
    f"{ullage*100:g}% ullage",
    fontsize=14, fontweight="bold"
)

plt.show()
