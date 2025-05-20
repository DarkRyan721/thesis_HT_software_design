from PySide6.QtCore import QThread, Signal
import os
import numpy as np
import pyvista as pv
import meshio

from magnetic_field_noGPU import B_Field

class LoaderWorker(QThread):
    finished = Signal(object)

    def __init__(self, mode="mesh", params = None):
        super().__init__()
        self.mode = mode
        self.params = params

    def run(self):
        if self.mode == "mesh":
            msh = meshio.read("./data_files/SimulationZone.msh")
            cells = msh.cells_dict.get("triangle")
            if cells is None:
                return
            faces = np.hstack([np.full((cells.shape[0], 1), 3), cells]).astype(np.int32).flatten()
            mesh = pv.PolyData(msh.points, faces)
            self.finished.emit(mesh)

        elif self.mode == "field":
            path = os.path.abspath("./data_files/Electric_Field_np.npy")
            if not os.path.exists(path):
                print("⚠️ Campo eléctrico no encontrado")
                return

            data = np.load(path)
            if data.shape[1] < 6:
                print("⚠️ Formato de campo eléctrico inválido")
                return

            points = data[:, :3]
            vectors = data[:, 3:]
            magnitudes = np.linalg.norm(vectors, axis=1)
            log_magnitudes = np.log10(magnitudes + 1e-3)

            mesh = pv.PolyData(points)
            mesh["vectors"] = vectors
            mesh["magnitude"] = log_magnitudes
            glyphs = mesh.glyph(orient="vectors", scale=False, factor=0.01)

            self.finished.emit(glyphs)

        elif self.mode == "magnetic":
            print("🔄 Magnetic field thread iniciado")
            nSteps, N, I = self.params
            magnetic_instance = B_Field(nSteps=nSteps, N=N, I=I)

            E_File = np.load("data_files/Electric_Field_np.npy")
            spatial_coords = E_File[:, :3]

            # ⚠️ Este cálculo pesado ahora se hace en el hilo secundario
            B_value = magnetic_instance.Total_Magnetic_Field(S=spatial_coords)
            magnetic_instance.Save_B_Field(B=B_value, S=spatial_coords)

            points = spatial_coords
            vectors = B_value
            magnitudes = np.linalg.norm(vectors, axis=1)

            mesh = pv.PolyData(points)
            mesh["vectors"] = vectors
            mesh["magnitude"] = magnitudes
            glyphs = mesh.glyph(orient="vectors", scale=False, factor=0.01)

            self.finished.emit(glyphs)

