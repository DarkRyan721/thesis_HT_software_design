import sys
import os



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from particle_in_cell_cpu import PIC
from electron_density_model import ElectronDensityModel
from electric_field_solver import ElectricFieldSolver
from magnetic_field_solver_cpu import B_Field
from mesh_generator import HallThrusterMesh
from project_paths import data_file, project_file

import time
import numpy as np
from dolfinx import fem, io
from tqdm import tqdm
from mpi4py import MPI
import traceback



# Aquí va el try/except con el bloque generador de archivos
try:
    print("Se generarán los archivos que faltan...")
    # From GUI inputs:
    outer_radius = 0.1/2 #0.1/2
    inner_radius = 0.056/2 #0.056/2
    height = 0.02 #0.02
    refinement = "test"
    mesh_gen = HallThrusterMesh(R_big=outer_radius, R_small=inner_radius, H=height, refinement_level=refinement)
    mesh_gen.generate()

    t_total_start = time.perf_counter()
    solver = ElectricFieldSolver()
    print("\n✅ Mesh loaded successfully!")

    # Solve Laplace equation
    Volt_input = 300
    print("\n⚡ Solving Laplace equation...")
    t_laplace_start = time.perf_counter()
    phi_laplace, E_laplace = solver.solve_laplace(Volt=Volt_input)
    solver.save_electric_field_numpy(E_laplace, filename="E_Field_Laplace.npy")
    t_laplace_end = time.perf_counter()
    print(f"✅ Laplace solution completed in {t_laplace_end - t_laplace_start:.2f} s.")

    with io.XDMFFile(MPI.COMM_WORLD, data_file("SimulationZone.xdmf"), "r") as xdmf:
        domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")

    model = ElectronDensityModel(domain)
    model.generate_density()
    model.save_density()

    # Solve Poisson equation
    print("\n🧪 Solving Poisson equation...")
    t_poisson_start = time.perf_counter()
    source_term = solver.load_density_from_npy()
    phi_poisson, E_poisson = solver.solve_poisson(source_term=source_term)
    solver.save_electric_field_numpy(E_poisson, filename="E_Field_Poisson.npy")
    t_poisson_end = time.perf_counter()
    print(f"✅ Poisson solution completed in {t_poisson_end - t_poisson_start:.2f} s.")

    t_total_end = time.perf_counter()
    print(f"\n🧾 Total execution time: {t_total_end - t_total_start:.2f} s.")
    print("🏁 Test Completed Successfully!")

    E_File = np.load(data_file("E_Field_Laplace.npy"))
    spatial_coords = E_File[:, :3]

    B_field = B_Field(nSteps=1500)

    B_value = B_field.Total_Magnetic_Field(S=spatial_coords)

    B_field.Save_B_Field(B=B_value, S=spatial_coords)

    N = 1000
    dt = 0.00000004
    q_m = 7.35e5
    alpha = 0.9
    frames = 500
    sigma_ion = 1e-11

    pic = PIC(Rin=0.028, Rex=0.05, N=N, L=0.02, dt=dt, q_m=q_m, alpha=alpha, sigma_ion=sigma_ion)
    pic.initizalize_to_simulation(v_neutro=200, timesteps=frames)
    pic.render()

except Exception as e:
    print("[ERROR] electric_field_process.py falló:")
    traceback.print_exc()
    # Opcional: escribe un archivo de error
    with open("electric_field_process_error.log", "w") as ferr:
        ferr.write(traceback.format_exc())
    sys.exit(1)



