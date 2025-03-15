import numpy as np

def aplicar_termostato(velocities, mass, N, dt, tau_T, T0, k_B):
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
    # Calculamos la energía cinética
    kinetic_energy = 0.5 * mass * np.sum(velocities**2)
    
    # Temperatura instantánea en 3D
    T_inst = (2.0 / (3.0 * k_B * N)) * kinetic_energy
    
    # Evitar inestabilidades numéricas:
    if T_inst > 1e-12:
        lambda_factor = np.sqrt(
            1.0 + (dt / tau_T) * ((T0 / T_inst) - 1.0)
        )
        velocities *= lambda_factor
    
    return velocities

