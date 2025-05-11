from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton, QSizePolicy
from styles.stylesheets import button_style, button_style_active


class OptionsPanel(QFrame):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setStyleSheet("background-color: #131313; border-radius: 0px;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.tab_buttons = []
        self.create_buttons()

    def create_buttons(self):
        btn_info = [
            ("Home", "🏠"),
            ("Mesh", "▩"),
            ("LaPlace", "E"),
            ("Magnet", "B"),
            ("Density", "σ"),
            ("Simulation", "⯈")
        ]

        for index, (name, label) in enumerate(btn_info):
            btn = QPushButton(label)
            btn.setObjectName(f"{name}_Btn")
            btn.setStyleSheet(button_style_active() if index == 0 else button_style())
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            btn.setMinimumSize(60, 10)
            btn.clicked.connect(lambda _, i=index, b=btn: self.main_window.on_dynamic_tab_clicked(i, b))
            self.layout.addWidget(btn)
            self.tab_buttons.append(btn)

        self.layout.addStretch()
