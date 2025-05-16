# utils/field_loader.py
from PySide6.QtCore import QThread, Signal
import numpy as np
import pyvista as pv
import os

class FieldLoaderWorker(QThread):
    finished = Signal(object)

    def __init__(self, filename="Electric_Field_np.npy", scale=0.01):
        super().__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.npy_path = os.path.join(base_dir, "..", "data_files", filename)
        self.scale = scale

    def run(self):
        if not os.path.exists(self.npy_path):
            print(f"❌ Archivo no encontrado: {self.npy_path}")
            return

        data = np.load(self.npy_path)
        if data.shape[1] < 6:
            print("❌ Formato inválido: se esperan columnas [x, y, z, Ex, Ey, Ez]")
            return

        points = data[:, :3]
        vectors = data[:, 3:]
        magnitudes = np.linalg.norm(vectors, axis=1)
        log_magnitudes = np.log10(magnitudes + 1e-3)

        mesh = pv.PolyData(points)
        mesh["vectors"] = vectors
        mesh["magnitude"] = log_magnitudes

        glyphs = mesh.glyph(orient="vectors", scale=False, factor=self.scale)
        self.finished.emit(glyphs)
