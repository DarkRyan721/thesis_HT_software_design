import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from models.simulation_state import SimulationState
from magnetic_field_solver_cpu import B_Field
import pyvista as pv

from project_paths import data_file, model

print("[DEBUG] process lanzado", flush=True)

E_File = np.load(data_file("E_Field_Laplace.npy"))
spatial_coords = E_File[:, :3]
state = SimulationState.load_from_json(model("simulation_state.json"))

nSteps = state.nSteps          # Ya es int, si lo puso así en el JSON
N_turns = state.N_turns        # Ya es int
I = state.I                    # Ya es float

bfield = B_Field(nSteps=nSteps, N=N_turns, I=I)
B_value = bfield.Magnetic_Field(S=spatial_coords, S_solenoid=bfield.S_Inner)
B_value = bfield.Total_Magnetic_Field(S=spatial_coords)

print("OK: Magnetic field calculated")
