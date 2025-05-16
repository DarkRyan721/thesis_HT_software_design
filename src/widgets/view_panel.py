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
from utils.mesh_loader import LoaderWorker


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

        self.current_data = None  # Útil para referenciar luego

    def update_viewer(self, data, viewer_name="mesh"):
        if viewer_name not in self.viewers:
            print(f"⚠️ Viewer '{viewer_name}' no registrado.")
            return

        self.current_data = data
        viewer = self.viewers[viewer_name]
        viewer.clear()
        viewer.add_mesh(data, color="white", show_edges=True)
        self.view_stack.setCurrentWidget(viewer)

        # Aplicar estilo si está en 'home_panel'
        if viewer_name == "mesh":
            print("⚠️ Aplicando estilo a home_viewer")
            QTimer.singleShot(0, lambda: self.main_window.home_panel.on_mesh_loaded())
        elif viewer_name == "field":
            print("⚠️ Aplicando estilo a field_viewer")
            QTimer.singleShot(0, lambda: self.main_window.field_panel.on_field_loaded())
            # Aplicar estilo si es necesario
        
    def switch_view(self, name: str):
        if name in self.viewers:
            self.view_stack.setCurrentWidget(self.viewers[name])
        else:
            print(f"⚠️ Viewer '{name}' no encontrado.")

    def update_mesh_viewer(self, mesh):
        self.update_viewer(mesh, viewer_name="mesh")

    def update_field_viewer(self, field):
        self.update_viewer(field, viewer_name="field")
