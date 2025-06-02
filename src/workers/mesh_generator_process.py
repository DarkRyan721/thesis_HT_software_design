import io
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import json
import traceback
import numpy as np

from project_paths import model
from models.simulation_state import SimulationState
from mesh_generator import HallThrusterMesh

print("[DEBUG] Lanzando run_mesher.py...")
try:

    state = SimulationState.load_from_json(model("simulation_state.json"))

    # Asigne las variables desde el estado
    H = state.H
    R_big = state.R_big
    R_small = state.R_small
    refinement_level = state.refinement_level
    min_physical_scale = state.min_physics_scale
    max_elements = state.max_elements
    print(refinement_level)
    mesh = HallThrusterMesh(
        R_big=R_big,
        R_small=R_small,
        H=H,
        refinement_level=refinement_level
    )
    mesh.generate()

except Exception as e:
    print("[ERROR] mesh process failed :")
    traceback.print_exc()
    with open("run_mesher_error.log", "w") as ferr:
        ferr.write(traceback.format_exc())
    sys.exit(1)
