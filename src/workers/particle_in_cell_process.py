import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import json
import traceback
import numpy as np

print("[DEBUG] Lanzando run_mesher.py...")

N = int(sys.argv[1])
frames = int(sys.argv[2])
# Lee argumentos avanzados si se pasan, si no usa los defaults
alpha = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 0.9
sigma_ion = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 1e-11
dt = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 0.00000004
q_m = 7.35e5
GPU_ACTIVE = sys.argv[6].lower() == "true"

if GPU_ACTIVE:
    from particle_in_cell import PIC
else:
    from particle_in_cell_cpu import PIC

gas = sys.argv[7] if len(sys.argv) > 7 else "Xenon"

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

try:
    print(f"[INFO] N={N}, frames={frames}, alpha={alpha}, sigma_ion={sigma_ion}, dt={dt}, q_m={q_m}, GPU_ACTIVE={GPU_ACTIVE}, gas='{gas}', V_neutro={V_neutro}")
    pic = PIC(Rin=0.028, Rex=0.05, N=N, L=0.02, dt=dt, q_m=q_m, alpha=alpha, sigma_ion=sigma_ion)
    pic.initizalize_to_simulation(v_neutro = V_neutro, timesteps=frames)
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
