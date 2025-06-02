import subprocess
import sys
import time
from PySide6.QtCore import QThread, Signal, QObject
import os
import numpy as np
import pyvista as pv
import meshio

from electric_field_solver import ElectricFieldSolver
from magnetic_field_solver_cpu import B_Field
from simulation_engine_viewer import Simulation  # importa tu clase
from PySide6.QtCore import Slot
from project_paths import data_file

class LoaderWorker(QObject):
    finished = Signal(object)
    started = Signal()
    progress = Signal(int)

    def __init__(self, mode="mesh", params = None, plotter=None, solver = None ):
        super().__init__()
        self.mode = mode
        self.params = params
        self.plotter = plotter
        self.solver = solver
        self._is_running = True

    @Slot()
    def run(self):
        # print(f"✅ Worker RUN llamado en modo: {self.mode}")

        if self.mode == "mesh":
            msh = meshio.read(data_file("SimulationZone.msh"))
            cells = msh.cells_dict.get("triangle")
            if cells is None:
                return
            faces = np.hstack([np.full((cells.shape[0], 1), 3), cells]).astype(np.int32).flatten()
            mesh = pv.PolyData(msh.points, faces)
            self.finished.emit(mesh)

        elif self.mode == "field":
            self.progress.emit(0)
            validate_density = self.params.get("validate_density")
            laplace_path = data_file("E_Field_Laplace.npy")
            poisson_path = data_file("E_Field_Poisson.npy")
            field_path = poisson_path if validate_density else laplace_path

            # Seleccionar archivo según el checkbox
            print(field_path)
            field_path = poisson_path if validate_density else laplace_path
            if not os.path.exists(field_path):
                print(f"⚠️ Archivo de campo no encontrado: {field_path}")
                return

            data = np.load(field_path)
            if data.shape[1] < 6:
                print("⚠️ Formato de campo eléctrico inválido")
                return

            points = data[:, :3]
            vectors = data[:, 3:]
            magnitudes = np.linalg.norm(vectors, axis=1)
            log_magnitudes = np.log10(magnitudes + 1e-3)
            log_magnitudes[log_magnitudes < 0] = 0

            mesh = pv.PolyData(points)
            mesh["E_field"] = vectors
            mesh["magnitude"] = magnitudes
            self.progress.emit(100)
            self.finished.emit(mesh)

        elif self.mode == "magnetic":
            print("🔄 Cargando campo magnético...")
            file_path = data_file("Magnetic_Field_np.npy")
            if not os.path.exists(file_path):
                print(f"⚠️ Archivo no encontrado: {file_path}")
                return

            data = np.load(file_path)
            if data.shape[1] < 6:
                print("⚠️ Formato de campo magnético inválido")
                return
            spatial_coords = data[:, :3]
            B_value = data[:, 3:]

            points = spatial_coords
            vectors = B_value
            magnitudes = np.linalg.norm(vectors, axis=1)

            mesh = pv.PolyData(points)
            mesh["vectors"] = vectors
            mesh["magnitude"] = magnitudes
            glyphs = mesh.glyph(orient="vectors", scale=False, factor=0.01)
            self.finished.emit(glyphs)

        elif self.mode == "density":
            try:

                # Cargar los datos de densidad directamente para emitirlos como PolyData
                E_np = np.load(data_file("E_Field_Laplace.npy"))
                points = E_np[:, :3]
                n0 = np.load(data_file("density_n0.npy"))
                n0_log = np.log10(n0)
                mesh = pv.PolyData(points)
                max_value = np.max(n0_log[n0_log != np.log10(1e-100)])
                log_min = max_value - 3.0
                log_max = max_value

                n0_log = np.log10(n0)
                mesh["n0_log"] = n0_log
                self.finished.emit({
                    "density": mesh,
                    "log_min": log_min,
                    "log_max": log_max
                })
            except Exception as e:
                print(f"❌ Error al cargar la densidad: {e}")

        elif self.mode == "simulation":
            print("🚀 Iniciando simulación de partículas...")
            if self.plotter is None:
                print("⚠️ Instancia de simulación no proporcionada.")
                return
            try:
                print("🔄 Ejecutando simulación...")
                self.plotter.Animation(neutral_visible=self.params.get("neutral_visible", False))
                self.finished.emit("Simulation completed")
            except Exception as e:
                print(f"❌ Error durante la simulación: {e}")


