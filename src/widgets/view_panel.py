import sys
import os
# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import PySide6.QtWidgets as QtW
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton, QSizePolicy
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtCore import QTimer
import meshio
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore
from styles.stylesheets import *
from utils.mesh_loader import MeshLoaderWorker


class ViewPanel(QFrame):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setObjectName("VPartFrame")
        self.setStyleSheet("background-color: #D3D3D3; border-left: 2px solid #818589;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # 🧱 Stack de vistas
        self.view_stack = QStackedWidget(self)
        layout.addWidget(self.view_stack)

        self.viewers = {
            "mesh": self.main_window.home_panel.home_viewer,
            "field": self.main_window.field_panel.field_viewer,
            # Puedes agregar más si lo necesitas
        }

        # Añadir viewers al stack
        for viewer in self.viewers.values():
            self.view_stack.addWidget(viewer)

        self.current_view = None  # Útil para referenciar luego

        # 🌐 Cargar malla de ejemplo
        self.worker = MeshLoaderWorker()
        self.worker.finished.connect(self.update_viewer)
        self.worker.start()

    def update_viewer(self, mesh, viewer_name="mesh"):
        if viewer_name not in self.viewers:
            print(f"⚠️ Viewer '{viewer_name}' no registrado.")
            return

        self.current_view = mesh
        viewer = self.viewers[viewer_name]
        viewer.clear()
        viewer.add_mesh(mesh, color="white", show_edges=True)
        self.view_stack.setCurrentWidget(viewer)

        # Aplicar estilo si está en 'home_panel'
        if viewer_name == "mesh":
            QTimer.singleShot(0, lambda: self.main_window.home_panel._apply_visual_style_home())
        elif viewer_name == "field":
            QTimer.singleShot(0, lambda: self.main_window.field_panel._apply_visual_style())

            # Aplicar estilo si es necesario
            QTimer.singleShot(0, lambda: self.main_window.home_panel._apply_visual_style())

    def switch_view(self, name: str):
        if name in self.viewers:
            self.view_stack.setCurrentWidget(self.viewers[name])
        else:
            print(f"⚠️ Viewer '{name}' no encontrado.")

    def update_mesh_viewer(self, mesh):
        self.update_viewer(mesh, viewer_name="mesh")

    def update_field_viewer(self, mesh):
        self.update_viewer(mesh, viewer_name="field")
