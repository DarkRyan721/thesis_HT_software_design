import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import json
import traceback
import numpy as np


from mesh_generator import HallThrusterMesh

print("[DEBUG] Lanzando run_mesher.py...")
try:
    H = float(sys.argv[1])
    R_big = float(sys.argv[2])
    R_small = float(sys.argv[3])
    refinement_level = sys.argv[4]
    min_physical_scale = float(sys.argv[5]) if sys.argv[5] else None
    max_elements = float(sys.argv[6]) if sys.argv[6] else None

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
