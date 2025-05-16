class SimulationState:
    def __init__(self):
        self.H = 0.02
        self.R_big = 0.050
        self.R_small = 0.027
        self.voltage = 300
        self.voltage_cathode = 16
        self.nSteps: int = 5000
        self.N: int = 200
        self.I: float = 4.5


        self.prev_params_mesh = (None, None, None)
        self.prev_params_field = (None, None)
        self.prev_params_magnetic = (None, None, None)

    def print_state(self):
        print(f"SimulationState(H={self.H}, R_big={self.R_big}, R_small={self.R_small}, "
              f"voltage={self.voltage}, voltage_cathode={self.voltage_cathode}, "
              f"nSteps={self.nSteps}, N={self.N}, I={self.I})")