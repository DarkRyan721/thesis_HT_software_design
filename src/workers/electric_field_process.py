import io
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from models.simulation_state import SimulationState
from electron_density_model import ElectronDensityModel
import traceback
from electric_field_solver import ElectricFieldSolver
from project_paths import data_file, model
from dolfinx import fem, io
from mpi4py import MPI

try:

    state = SimulationState.load_from_json(model("simulation_state.json"))

    volt = state.voltage
    volt_cath = state.voltage_cathode


    solver = ElectricFieldSolver()

        # --- LAPLACE ---
    print("[DEBUG] Resolviendo Laplace...")
    _, E_laplace = solver.solve_laplace(
        Volt=volt,
        Volt_cath=volt_cath
    )
    output_file_laplace = data_file("E_Field_Laplace.npy")
    solver.save_electric_field_numpy(E_laplace, filename=output_file_laplace)
    print(f"[DEBUG] Guardado Laplace en: {output_file_laplace}")

    with io.XDMFFile(MPI.COMM_WORLD, data_file("SimulationZone.xdmf"), "r") as xdmf:
        domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")

    model = ElectronDensityModel(domain)
    model.generate_density()
    model.save_density()

    # --- POISSON ---
    print("[DEBUG] Resolviendo Poisson...")
    source_term = solver.load_density_from_npy()
    _, E_poisson = solver.solve_poisson(
        source_term=source_term,
        Volt=volt,
        Volt_cath=volt_cath
    )
    output_file_poisson = data_file("E_Field_Poisson.npy")
    solver.save_electric_field_numpy(E_poisson, filename=output_file_poisson)
    print(f"[DEBUG] Guardado Poisson en: {output_file_poisson}")


    print("[DEBUG] run_solver.py terminó exitosamente.")

except Exception as e:
    print("[ERROR] run_solver.py falló:")
    traceback.print_exc()
    with open("electric_field_process_error.log", "w") as ferr:
        ferr.write(traceback.format_exc())
    sys.exit(1)
