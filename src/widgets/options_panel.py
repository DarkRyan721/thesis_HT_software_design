import sys
import os
# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import PySide6.QtWidgets as QtW
from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton, QSizePolicy
from styles.stylesheets import *


class OptionsPanel(QFrame):
    def __init__(self, main_window, parameters_view):
        super().__init__()
        self.main_window = main_window
        self.parameters_view = parameters_view
        self.parameters_view.setCurrentIndex(0)

        self.setStyleSheet("background-color: #131313; border-radius: 0px;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.tab_buttons = []
        self.create_buttons()
        self.create_option_panels()

    def create_buttons(self):
        btn_info = [
            ("Home", "🏠"),
            ("LaPlace", "E"),
            ("Magnet", "B"),
            ("Density", "σ"),
            ("Simulation", "⯈")
        ]

        for index, (name, label) in enumerate(btn_info):
            btn = QPushButton(label)
            btn.setObjectName(f"{name}_Btn")
            btn.setStyleSheet(button_activate_style() if index == 0 else button_options_style())
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            btn.setMinimumSize(60, 10)
            btn.clicked.connect(lambda _, i=index, b=btn: self.main_window.on_dynamic_tab_clicked(i, b))
            self.layout.addWidget(btn)
            self.tab_buttons.append(btn)

        self.layout.addStretch()

    def create_option_panels(self):
        self.parameters_view.addWidget(self.main_window.home_panel)        # index 0
        self.parameters_view.addWidget(self.main_window.field_panel)     # index 2
        self.parameters_view.addWidget(self.main_window.magnetic_panel)      # index 3
        self.parameters_view.addWidget(self.main_window.Density_Options())     # index 4
        self.parameters_view.addWidget(self.main_window.Simulation_Options())  # index 5

