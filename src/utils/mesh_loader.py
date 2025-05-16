from PySide6.QtCore import QThread, Signal
import os
import numpy as np
import pyvista as pv
import meshio

class LoaderWorker(QThread):
    finished = Signal(object)

    def __init__(self, mode="mesh"):
        super().__init__()
        self.mode = mode

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

