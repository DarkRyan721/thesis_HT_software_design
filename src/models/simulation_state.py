import json

class SimulationState:
    # ... tus atributos y métodos ...
    def to_dict(self):
        return self.__dict__

    def save_to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls()
        obj.__dict__.update(data)
        return obj
