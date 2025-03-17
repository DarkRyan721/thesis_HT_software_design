import numpy as np

def aplicar_termostato(kinetic_energy, N, dt, tau_T, T0, k_B):
    """
    Aplica el termostato a las velocidades para llevar la temperatura al valor T0.
    
    velocities: arreglo (N,3) de velocidades de N partículas
    mass: masa de cada partícula (puede ser un float, si todas tienen la misma masa)
    N: número de partículas
    dt: paso de tiempo
    tau_T: tiempo de relajación del termostato
    T0: temperatura objetivo
    k_B: constante de Boltzmann
    """
    # Temperatura instantánea en 3D
    T_inst = (2.0 / (3.0 * k_B * N)) * kinetic_energy
    
    # Evitar inestabilidades numéricas:
    #if T_inst > 1e-12:
    lambda_factor = np.sqrt(
        1.0 + (dt / tau_T) * ((T0 / T_inst) - 1.0)
    )
    lambda_factor

    Temp = np.mean(T_inst).item()
    
    return lambda_factor, Temp

