import sys
import os
from PySide6.QtWidgets import QApplication, QSplashScreen, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QTimer, QProcess
from main_window import MainWindow
import signal

from project_paths import data_file, project_file, worker

class LoadingScreen(QWidget):
    def __init__(self, msg):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
        layout = QVBoxLayout(self)
        self.label = QLabel(msg)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.resize(400, 100)
        self.setWindowTitle("Preparando datos...")

def run_app():
    # ... tu lógica para scripts y archivos ...
    script_files = [
        "electric_field_solver.py",
        "magnetic_field_solver_cpu.py",
        "mesh_generator.py",
        "project_paths.py",
        "particle_in_cell_cpu.py"
    ]

    data_files = [
        "SimulationZone.msh",
        "E_Field_Laplace.npy",
        "E_Field_Poisson.npy",
        "Magnetic_Field_np.npy",
        "particle_simulation.npy"
    ]

    missing_scripts = [f for f in script_files if not os.path.isfile(project_file(f))]
    missing_data = [f for f in data_files if not os.path.isfile(data_file(f))]

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if missing_scripts or missing_data:
        loading_screen = LoadingScreen("Generando archivos iniciales. Por favor, espere...")
        loading_screen.show()

        process = QProcess()
        process.setProgram(sys.executable)
        process.setArguments([worker("initial_state_process.py")])

        def check_finished(exitCode, exitStatus):
            loading_screen.label.setText(f"Generación finalizada. Código: {exitCode}")
            loading_screen.hide()
            window = MainWindow()
            window.show()
            # Ya puedes cerrar loading_screen
            # Si quieres, elimina loading_screen: loading_screen.deleteLater()

        process.finished.connect(check_finished)
        process.start()

        # Lanzar la app (esto mantiene vivo el event loop)
        sys.exit(app.exec())

    else:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
