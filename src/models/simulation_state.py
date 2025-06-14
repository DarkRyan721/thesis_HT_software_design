import json


class SimulationState:
    def __init__(self):
        self.H = 0.02
        self.R_big = 0.050
        self.R_small = 0.028
        self.refinement_level = "test"
        self.voltage = 300
        self.voltage_cathode = 16
        self.nSteps: int = 5000
        self.N_turns: int = 200
        self.I: float = 4.5
        self.r0 = (self.R_big + self.R_small)/2
        self.N_particles = 1000
        self.frames = 500

        self.prev_params_mesh = (None, None, None, None, None)
        self.prev_params_field = (self.voltage, self.voltage_cathode)
        self.prev_params_magnetic = (None, None, None)
        self.prev_params_simulation = (None, None)

        self.min_physics_scale = None
        self.max_elements = None

        self.chunk_size = None

        self.alpha = None
        self.sigma_ion = None
        self.dt = None


    def print_state(self):
        print(f"SimulationState(H={self.H}, R_big={self.R_big}, R_small={self.R_small}, "
            f"voltage={self.voltage}, voltage_cathode={self.voltage_cathode}, "
            f"nSteps={self.nSteps}, N={self.N_turns}, I={self.I})")

    # ... tus atributos y métodos ...
    def to_dict(self):
        return self.__dict__

    def save_to_json(self, path):
        print(f"[DEBUG] Guardando JSON en: {path}")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls()
        obj.__dict__.update(data)
        return obj

