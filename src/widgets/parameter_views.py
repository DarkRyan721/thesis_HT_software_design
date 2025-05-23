import sys
import os
# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import PySide6.QtWidgets as QtW
from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton, QSizePolicy
from PySide6.QtCore import Qt, QThread, Signal
from gui_styles.stylesheets import *


class ParameterPanel(QFrame):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setStyleSheet("background-color: #fbfcfc; border-radius: 0px;")
        self.setFixedWidth(640)
        self.setSizePolicy(QtW.QSizePolicy.Policy.Fixed, QtW.QSizePolicy.Policy.Expanding)

        parameters_layout = QVBoxLayout(self)
        parameters_layout.setContentsMargins(0, 5, 5, 5)

        self.parameters_view = QtW.QStackedWidget()
        self.parameters_view.setObjectName("Parameters_Views")
        self.parameters_view.setStyleSheet("""
        QStackedWidget#Parameters_Views {
            background-color: #2c2c2c;
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }
        """)
        
        parameters_layout.addWidget(self.parameters_view)
