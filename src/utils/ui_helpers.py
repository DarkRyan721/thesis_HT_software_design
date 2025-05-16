from PySide6.QtWidgets import (
    QWidget, QLineEdit, QLabel, QHBoxLayout
)

def _input_with_unit(default_value, unit):
    field = QLineEdit(default_value)
    unit_lbl = QLabel(unit)
    unit_lbl.setStyleSheet("color: gray; padding-left: 5px;")
    box = QHBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(field)
    box.addWidget(unit_lbl)

    container = QWidget()
    container.setLayout(box)
    return container, field