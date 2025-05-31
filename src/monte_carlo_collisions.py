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
