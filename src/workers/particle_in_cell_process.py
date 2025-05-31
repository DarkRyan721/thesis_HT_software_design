import io
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from particle_in_cell_cpu import PIC
import json
import traceback
import numpy as np

print("[DEBUG] Lanzando run_mesher.py...")

N = int(sys.argv[1])
frames = int(sys.argv[2])
dt = 0.00000004
q_m = 7.35e5
alpha = 0.9
sigma_ion = 1e-11
try:
        # def leer_datos_archivo(ruta_archivo):
        #     datos = {}
        #     with open(ruta_archivo, "r") as archivo:
        #         for linea in archivo:
        #             # Verificamos que la línea contenga ':'
        #             if ":" in linea:
        #                 clave, valor = linea.split(":", maxsplit=1)
        #                 # Limpiamos espacios
        #                 clave = clave.strip()
        #                 valor = valor.strip()
        #                 # Almacenar en el diccionario (conversión a entero o float)
        #                 datos[clave] = float(valor)
        #     return datos
        # ruta = data_file("geometry_parameters.txt")
        # info = leer_datos_archivo(ruta)

        # Rin = info.get("radio_interno",0) # Radio interno del cilindro hueco
        # Rex = info.get("radio_externo",0) # Primer radio externo del cilindro hueco
        # L = info.get("profundidad",0) # Longitud del cilindro

        # EXAMPLE FOR GUI:

    pic = PIC(Rin=0.028, Rex=0.05, N=N, L=0.02, dt=dt, q_m=q_m, alpha=alpha, sigma_ion=sigma_ion)

    # 2. Inicializar la simulacion(Posiciones iniciales, velocidades iniciales...)

    pic.initizalize_to_simulation(v_neutro=200, timesteps=frames)

    # 3. Realizar el render

    pic.render()


    import json

    # Supón que calculaste esto:
    impulso_especifico = pic.specific_impulse    # s

    resultados = {
        "impulso_especifico": pic.specific_impulse*0.2,
    }

    with open("resultados_simulacion.json", "w") as f:
        json.dump(resultados, f)

except Exception as e:
    print("[ERROR] mesh process failed :")
    traceback.print_exc()
    with open("run_mesher_error.log", "w") as ferr:
        ferr.write(traceback.format_exc())
    sys.exit(1)
