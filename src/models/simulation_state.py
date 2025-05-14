class SimulationState:
    def __init__(self):
        self.H = 0.02
        self.R_big = 0.050
        self.R_small = 0.027
        self.voltage = 300
        self.voltage_cathode = 16

        self.prev_params_mesh = (None, None, None)
        self.prev_params_field = (None, None)
