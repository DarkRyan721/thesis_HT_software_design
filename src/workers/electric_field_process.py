# run_solver.py
import sys
import json
import traceback

print("[DEBUG] Lanzando run_solver.py...")
print(f"[DEBUG] sys.argv: {sys.argv}")

try:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    print(f"[DEBUG] Leyendo parámetros desde: {input_file}")

    with open(input_file, 'r') as f:
        params = json.load(f)
    print(f"[DEBUG] Parámetros recibidos: {params}")

    from electric_field_solver import ElectricFieldSolver

    solver = ElectricFieldSolver()
    if params.get("validate_density", False):
        print("[DEBUG] Resolviendo Poisson...")
        source_term = solver.load_density_from_npy()
        _, E = solver.solve_poisson(
            source_term=source_term,
            Volt=params["voltage"],
            Volt_cath=params["voltage_cathode"]
        )
    else:
        print("[DEBUG] Resolviendo Laplace...")
        _, E = solver.solve_laplace(
            Volt=params["voltage"],
            Volt_cath=params["voltage_cathode"]
        )
    print(f"[DEBUG] Guardando resultado en: {output_file}")
    solver.save_electric_field_numpy(E, filename=output_file)
    print(f"[DEBUG] run_solver.py terminó exitosamente.")

except Exception as e:
    print("[ERROR] electric_field_process.py falló:")
    traceback.print_exc()
    # Opcional: escribe un archivo de error
    with open("electric_field_process_error.log", "w") as ferr:
        ferr.write(traceback.format_exc())
    sys.exit(1)
