import io
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from project_paths import model
from models.simulation_state import SimulationState
import json
import traceback
import numpy as np

print("[DEBUG] Lanzando run_mesher.py...")

state = SimulationState.load_from_json(model("simulation_state.json"))

# Parámetros principales
N = getattr(state, "N_particles", 1000)      # Default = 1000
frames = getattr(state, "frames", 500)       # Default = 500

# Avanzados (con defaults si no existen)
alpha = getattr(state, "alpha", None)
if alpha is None:
    alpha = 0.9

sigma_ion = getattr(state, "sigma_ion", None)
if sigma_ion is None:
    sigma_ion = 1e-11

dt = getattr(state, "dt", None)
if dt is None:
    dt = 4e-8

# GPU activo
GPU_ACTIVE = getattr(state, "GPU_ACTIVE", False)

if GPU_ACTIVE:
    from particle_in_cell import PIC
else:
    from particle_in_cell_cpu import PIC

# Tipo de gas y propiedades asociadas
gas = getattr(state, "gas", "Xenon")

if gas == "Xenon":
    q_m = 7.35e5
    V_neutro = 200
elif gas == "Krypton":
    q_m = 11.5e5
    V_neutro = 220
elif gas == "Argon":
    q_m = 24.2e5
    V_neutro = 430
elif gas == "Helium":
    q_m = 41.9e5
    V_neutro = 1100
else:
    q_m = 7.35e5
    V_neutro = 200

# (Opcional) Imprima para verificar
print(f"N={N}, frames={frames}, alpha={alpha}, sigma_ion={sigma_ion}, dt={dt}, GPU={GPU_ACTIVE}, gas={gas}")
print(f"q/m={q_m}, V_neutro={V_neutro}")

try:
    Rin = state.R_small      # o el nombre correcto que tenga en el JSON, por ejemplo "R_small"
    Rex = state.R_big      # o "R_big", según su estructura
    L = state.H

    pic = PIC(
        Rin=Rin,
        Rex=Rex,
        N=N,
        L=L,
        dt=dt,
        q_m=q_m,
        alpha=alpha,
        sigma_ion=sigma_ion
    )
    pic.initizalize_to_simulation(v_neutro=V_neutro, timesteps=frames)
    pic.render()

    impulso_especifico = pic.specific_impulse

    resultados = {
        "impulso_especifico": impulso_especifico * 0.2,
    }
    with open("resultados_simulacion.json", "w") as f:
        json.dump(resultados, f)

except Exception as e:
    print("[ERROR] mesh process failed :")
    traceback.print_exc()
    with open("run_mesher_error.log", "w") as ferr:
        ferr.write(traceback.format_exc())
    sys.exit(1)
