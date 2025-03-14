import numpy as np

k_B = 1.380649e10^23 #J/K
T0 = 300 #Temperatura objetivo en kelvin
tau_T = 100 #Tiempo de relajacion del termostato
dt = 0.01              # Paso de integración


"""
Calcula la temperatura instantánea (3D, todas las partículas con la misma masa).
"""
#mass es la masa de la particula
#N es el numero de particulas
#Velocities 

kinetic_energy = 0.5 * mass * np.sum(velocities**2)
# En 3D, grados de libertad totales ~ 3N (si no hay restricciones)
T_inst = (2.0 / (3.0 * k_B * N)) * kinetic_energy


if T_inst > 1e-12:  # Para evitar inestabilidades numéricas
    lambda_factor = np.sqrt(
        1.0 + (dt / tau_T) * ((T0 / T_inst) - 1.0)
    )
    v *= lambda_factor

