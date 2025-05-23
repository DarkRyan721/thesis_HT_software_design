import sys
import os
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
import signal
import subprocess

from project_paths import worker

def run_generation_subprocess():
    # Llama a tu script de verificación/generación
    # Puedes ajustar el path y el comando según tu estructura
    process = subprocess.Popen([sys.executable, worker("initial_state_process.py")])
    return process

def run_app():
    # Lanza el subproceso y espera a que termine ANTES de lanzar la interfaz
    gen_proc = run_generation_subprocess()
    print("Esperando a que termine el proceso de inicialización...")
    return_code = gen_proc.wait()
    print(f"Subproceso finalizado con código: {return_code}")

    # Si el subproceso terminó correctamente, lanza la app
    if return_code == 0:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    else:
        print("Error: El subproceso de generación falló. No se lanzará la interfaz.")
        sys.exit(1)

if __name__ == "__main__":
    run_app()
