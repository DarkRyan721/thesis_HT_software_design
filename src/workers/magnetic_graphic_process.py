# magnetic_graphic_process.py
import os
import sys
import json


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from models.simulation_state import SimulationState
from project_paths import model

from magnetic_field_solver_cpu import B_Field
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":

    state = SimulationState.load_from_json(model("simulation_state.json"))


    # Extraer los parámetros
    mode = state.mode.lower()
    nSteps = int(state.nSteps)
    N_turns = int(state.N_turns)
    I = float(state.I)
    output_file = state.output_file
    kwargs = state.kwargs

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