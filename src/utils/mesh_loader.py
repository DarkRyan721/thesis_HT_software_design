from PySide6.QtCore import QThread, Signal
import meshio
import numpy as np
import pyvista as pv
from styles.stylesheets import *

class MeshLoaderWorker(QThread):
    finished = Signal(object)

    def __init__(self):
        super().__init__()
        self.msh = meshio.read("./data_files/SimulationZone.msh")

    def run(self):
        cells = self.msh.cells_dict.get("triangle")
        if cells is None:
            return

        num_cells = cells.shape[0]
        faces = np.hstack([np.full((num_cells, 1), 3), cells]).astype(np.int32).flatten()
        mesh = pv.PolyData(self.msh.points, faces)

        self.finished.emit(mesh)
