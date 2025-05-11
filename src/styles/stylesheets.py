def checkbox_style(self):
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

    QCheckBox::indicator:hover {
        border: 1px solid #aaaaaa;
    }
    """

def button_style(self):
    return """
        QPushButton {
            background-color: #6082B6;
            color: white;
            font-weight: bold;
            padding: 6px 12px;
            border-radius: 6px;
            margin-top: 15px;
        }
        QPushButton:hover {
            background-color: #5c6bc0;
        }
        QPushButton:pressed {
            background-color: #303f9f;
        }
        QPushButton:disabled {
            background-color: #9e9e9e;
            color: #eeeeee;
        }
    """

def self_Style(self):
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

def button_activate_style(self):
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

def button_style(self):
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
