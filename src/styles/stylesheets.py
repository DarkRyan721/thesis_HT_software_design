def checkbox_style():
    return """
    QCheckBox {
        spacing: 8px;
        color: white;
        font-weight: normal;
    }

    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #888;
        border-radius: 4px;
        background: #2b2b2b;
    }

    QCheckBox::indicator:checked {
        background-color: #5c6bc0;
        border: 1px solid #5c6bc0;
    }

    QCheckBox::indicator:hover {a
        border: 1px solid #aaaaaa;
    }
    """

def button_parameters_style():
    return """
        QPushButton {
            background-color: #4a90e2;
            color: white;
            font-weight: 600;
            font-size: 13px;
            padding: 4px 10px;
            border-radius: 4px;
            margin-top: 10px;
            border: 1px solid #3a70b2;
        }
        QPushButton:hover {
            background-color: #5fa1f2;
        }
        QPushButton:pressed {
            background-color: #2e6bb2;
        }
        QPushButton:disabled {
            background-color: #b0b0b0;
            color: #f0f0f0;
            border: 1px solid #a0a0a0;
        }
    """

def self_Style():
    return """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 10px;
            padding: 5px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 3px;
            color: white;
        }

        QLabel {
            color: white;
        }

        QLineEdit {
            background-color: #1e1e1e;
            color: white;
            border: 1px solid #555;
            padding: 2px;
            border-radius: 3px;
        }
        QFrame#VPartFrame {
            background-color: #D3D3D3;
            border-left: 2px solid #818589;
            border-radius: 0px;
        }
    """

def button_activate_style():
    return """
    QPushButton {
        background-color: #2c2c2c;
        color: #ffffff;
        font-size: 30px;
        margin-top: 10px;
        font-weight: bold;
        border-radius: 10px;
        text-align: center;
        padding: 5px;  /* afecta el contenido interno */
        margin-left: 5px;  /* esto sí separa del borde */
        margin-right: 0;
        border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
    }"""

def button_options_style():
    return """
            QPushButton {
                background-color: #131313;
                color: #ffffff;
                font-size: 30px;
                margin-top: 10px;
                font-weight: bold;
                border-radius: 10px;
                text-align: center;
                padding: 5px;  /* afecta el contenido interno */
                margin-left: 5px;  /* esto sí separa del borde */
                margin-right: 5px;
            }

            QPushButton:hover {
                background-color: #ffffff;
                color: #131313;
                padding: 5px;  /* afecta el contenido interno */
                margin-left: 5px;  /* esto sí separa del borde */
                margin-right: 5px;
            }
            QPushButton:pressed {
                background-color: #212121;
                color: #131313;
                padding: 5px;  /* afecta el contenido interno */
                margin-left: 5px;  /* esto sí separa del borde */
                margin-right: 5px;
            }

            QPushButton:disabled {
                background-color: #131313;
                color: #393939;
                padding: 5px;  /* afecta el contenido interno */
                margin-left: 5px;  /* esto sí separa del borde */
                margin-right: 5px;
            }
        """
def box_render_style():
    return """
            QComboBox {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: white;
                selection-background-color: #5c6bc0;
                border: none;
            }
        """