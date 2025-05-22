from scipy.spatial import cKDTree
import numpy as np

from scipy.spatial import cKDTree

def MCC(s, v, s_label, rho, sigma_ion, dt, tree):
    """
    Versión vectorizada y optimizada del proceso de ionización.

    S, v, s_label deben ser CuPy arrays.
    s_rho y rho deben estar en NumPy (malla original), como se usan en KDTree.
    """
    _, idxs = tree.query(s.get())

    rho_part = rho[idxs]

    n_e = rho_part

    n_e_cp = cp.asarray(n_e)
    v_mag = cp.linalg.norm(v, axis=1)

    nu_ion = n_e_cp * sigma_ion * v_mag
    P_ion = 1 - cp.exp(-nu_ion * dt)

    aleatorios = cp.random.rand(len(s))
    ocurren = aleatorios < P_ion

    s_label[ocurren] = 1 - s_label[ocurren]

    return s_label

def MCC_Numpy(s, v, s_label, rho, sigma_ion, dt, tree):
    """
    Versión vectorizada y optimizada del proceso de ionización.

    S, v, s_label deben ser CuPy arrays.
    s_rho y rho deben estar en NumPy (malla original), como se usan en KDTree.
    """
    # Buscar el nodo más cercano para cada partícula
    _, idxs = tree.query(s)

    # Interpolar la densidad de electrones
    n_e = rho[idxs]

    # Calcular magnitud de velocidad para cada partícula
    v_mag = np.linalg.norm(v, axis=1)

    # Frecuencia de colisión e ionización
    nu_ion = n_e * sigma_ion * v_mag
    P_ion = 1 - np.exp(-nu_ion * dt)

    # Determinar si ocurre una ionización
    aleatorios = np.random.rand(len(s))
    ocurren = aleatorios < P_ion

    # Cambiar etiqueta 0 -> 1 o 1 -> 0 si hay cambio
    s_label[ocurren] = 1 - s_label[ocurren]

    return s_label