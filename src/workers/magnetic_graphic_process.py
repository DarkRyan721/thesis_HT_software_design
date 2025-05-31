# magnetic_graphic_process.py
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from magnetic_field_solver_cpu import B_Field
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("ERROR: Faltan argumentos.")
        sys.exit(1)

    mode = sys.argv[1].lower()
    nSteps = int(sys.argv[2])
    N_turns = int(sys.argv[3])
    I = float(sys.argv[4])
    output_file = sys.argv[5]
    kwargs = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}

    bfield = B_Field(nSteps=nSteps, N=N_turns, I=I)
    if mode == "fieldlines":
        fig, ax = bfield.B_Field_Lines(**kwargs)
    elif mode == "heatmap":
        fig, ax = bfield.B_Field_Heatmap(**kwargs)
    elif mode == "solenoid_points":
        fig, ax = bfield.Solenoid_points_plot(**kwargs)
    else:
        print(f"ERROR: Modo desconocido: {mode}")
        sys.exit(1)

    fig.savefig(output_file, dpi=150)
    print(f"OK: generated {output_file}")
    sys.exit(0)