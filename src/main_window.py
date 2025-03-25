import PySide6.QtWidgets as QtW
from PySide6.QtCore import Qt

class MainWindow(QtW.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HET simulator")
        self.resize(1100, 800)

        self.bool_hide_options = False

        self._setup_ui()

    def _setup_ui(self):
        #_____________________________________________________________________________________________
        #                   Ventana principal

        central_widget = QtW.QWidget() #Frame base de aplicacion Qt

        self.frame = QtW.QHBoxLayout()  #Frame principal. Contiene las tres columnas visibles
        self.frame.setSpacing(0) # Espaciado cero entre objetos
        self.frame.setContentsMargins(0,0,0,0) # Las columnas no tienen margen

        #_____________________________________________________________________________________________
        #                   Creacion de las tres columnas principales

        self.Options = self.set_Options()

        self.Parameters = self.set_Parameters()

        self.View_Part = self.set_View_Part()


        self.frame.addWidget(self.Options, stretch=0.3)
        self.frame.addWidget(self.Parameters, stretch=1)
        self.frame.addWidget(self.View_Part, stretch=2)
        

        #_____________________________________________________________________________________________
        #                   Añadiendo las configuraciones al frame base y frame principal

        central_widget.setLayout(self.frame)
        self.setCentralWidget(central_widget)

    def set_Options(self):
        style_btn = """
            QPushButton {
                background-color: #131313;
                color: #ffffff;
                font-size: 30px;
                margin-top: 10px;
                font-weight: bold;
                border-radius: 10px;
                text-align: center
            }
            
            QPushButton:hover {
                background-color: #ffffff;
                color: #131313;
            }
            
            QPushButton:pressed {
                background-color: #212121;
                color: #131313;
            }

            QPushButton:disabled {
                background-color: #131313;
                color: #393939;
            }
        """

        #_____________________________________________________________________________________________
        #                   Creacion del frame base y layout vertical para [Options]

        frame_options = QtW.QFrame()
        frame_options.setStyleSheet("background-color: #131313; border-radius: 0px;")

        options_layout = QtW.QVBoxLayout()

        #_____________________________________________________________________________________________
        #                   Creacion de componentes

        self.Home_Btn = QtW.QPushButton("🏠")
        self.Home_Btn.clicked.connect(self.Home_Options)
        self.Home_Btn.setStyleSheet(style_btn)
        self.Home_Btn.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Preferred)
        self.Home_Btn.setMinimumSize(50, 50)

        self.Mesh_Btn = QtW.QPushButton("▩")
        self.Mesh_Btn.clicked.connect(self.Mesh_Options)
        self.Mesh_Btn.setStyleSheet(style_btn)
        self.Mesh_Btn.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Preferred)
        self.Mesh_Btn.setMinimumSize(50, 50)

        self.LaPlace_Btn = QtW.QPushButton("E")
        self.LaPlace_Btn.clicked.connect(self.LaPlace_Options)
        self.LaPlace_Btn.setStyleSheet(style_btn)
        self.LaPlace_Btn.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Preferred)
        self.LaPlace_Btn.setMinimumSize(50, 50)
        self.LaPlace_Btn.setEnabled(False)

        self.MField_Btn = QtW.QPushButton("B")
        self.MField_Btn.clicked.connect(self.MField_Options)
        self.MField_Btn.setStyleSheet(style_btn)
        self.MField_Btn.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Preferred)
        self.MField_Btn.setMinimumSize(50, 50)
        self.MField_Btn.setEnabled(False)

        self.Density_Btn = QtW.QPushButton("σ")
        self.Density_Btn.clicked.connect(self.Density_Options)
        self.Density_Btn.setStyleSheet(style_btn)
        self.Density_Btn.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Preferred)
        self.Density_Btn.setMinimumSize(50, 50)
        self.Density_Btn.setEnabled(False)

        self.Simulation_Btn = QtW.QPushButton("⯈")
        self.Simulation_Btn.clicked.connect(self.Simulation_Options)
        self.Simulation_Btn.setStyleSheet(style_btn)
        self.Simulation_Btn.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Preferred)
        self.Simulation_Btn.setMinimumSize(50, 50)
        self.Simulation_Btn.setEnabled(False)

        #_____________________________________________________________________________________________
        #                   Añadiendo los componentes al Frame

        options_layout.addWidget(self.Home_Btn)
        options_layout.addWidget(self.Mesh_Btn)
        options_layout.addWidget(self.LaPlace_Btn)
        options_layout.addWidget(self.MField_Btn)
        options_layout.addWidget(self.Density_Btn)
        options_layout.addWidget(self.Simulation_Btn)
        options_layout.addStretch()
        frame_options.setLayout(options_layout)

        return frame_options

    def set_Parameters(self):
        #_____________________________________________________________________________________________
        #                   Creacion del frame base y layout vertical para [Options]

        frame_parameters = QtW.QFrame()
        frame_parameters.setStyleSheet("background-color: #212121; border-radius: 0px;")

        parameters_layout = QtW.QVBoxLayout()

        #_____________________________________________________________________________________________
        #                   Creacion de componentes

        self.Hide_Btn = QtW.QPushButton("≡ Ocultar")
        self.Hide_Btn.clicked.connect(self.Hide_Options)
        self.Hide_Btn.setStyleSheet("background-color: #131313;  color: white;  ")

        #_____________________________________________________________________________________________
        #                   Añadiendo los componentes al Frame

        #parameters_layout.addWidget(self.Hide_Btn)
        parameters_layout.addStretch()
        frame_parameters.setLayout(parameters_layout)

        return frame_parameters

    def set_View_Part(self):
        #_____________________________________________________________________________________________
        #                   Creacion del frame base y layout vertical para [Options]

        frame_VPart = QtW.QFrame()
        frame_VPart.setStyleSheet("background-color: #fbfcfc; border-radius: 0px;")

        VPart_layout = QtW.QVBoxLayout()

        #_____________________________________________________________________________________________
        #                   Creacion de componentes

        frame_Visual = QtW.QFrame()
        frame_Visual.setStyleSheet("background-color: #131313; border-radius: 5px;")
        frame_Visual.setFixedSize(550,350)
        
        # Contenedor horizontal para centrar frame_Visual
        h_container = QtW.QWidget()
        h_layout = QtW.QHBoxLayout()
        h_layout.addStretch()  # Espacio flexible a la izquierda
        h_layout.addWidget(frame_Visual)  # Widget centrado
        h_layout.addStretch()  # Espacio flexible a la derecha
        h_container.setLayout(h_layout)

        # Añadir el contenedor centrado al layout vertical
        VPart_layout.addWidget(h_container)
        frame_VPart.setLayout(VPart_layout)

        return frame_VPart
    
    def Home_Options(self):
        print("Home")

    def Mesh_Options(self):
        print("Mesh")

    def LaPlace_Options(self):
        print("LaPlace")

    def MField_Options(self):
        print("MField")

    def Density_Options(self):
        print("Density")

    def Simulation_Options(self):
        print("Simulation")

    def Hide_Options(self):
        self.bool_hide_options = not self.bool_hide_options

        if self.bool_hide_options == True:
            self.frame.setStretch(1,0.8)
            self.Hide_Btn.setText("≡")
        else:
            #self.centralWidget().layout().setStretch(0,1)
            self.frame.setStretch(1,1)
            self.Hide_Btn.setText("≡ Ocultar")

    def _on_button_click(self):
        self.label.setText("¡Botón presionado!")
        print("Evento ejecutado correctamente")