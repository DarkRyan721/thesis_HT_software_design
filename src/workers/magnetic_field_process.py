import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from magnetic_field_solver_cpu import B_Field
import pyvista as pv

from project_paths import data_file

print("[DEBUG] mesh_generator_process.py lanzado", flush=True)
print("Args:", sys.argv)

if len(sys.argv) < 4:
    print("[ERROR] Not enough arguments")
    sys.exit(1)

E_File = np.load(data_file("E_Field_Laplace.npy"))
spatial_coords = E_File[:, :3]
nSteps = int(sys.argv[1])
N_turns = int(sys.argv[2])
I = float(sys.argv[3])

bfield = B_Field(nSteps=nSteps, N=N_turns, I=I)
B_value = bfield.Magnetic_Field(S=spatial_coords, S_solenoid=bfield.S_Inner)
B_value = bfield.Total_Magnetic_Field(S=spatial_coords)

print("OK: Magnetic field calculated")
