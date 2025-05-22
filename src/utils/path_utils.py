# utils/path_utils.py
import os

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def get_data_file(filename):
    return os.path.join(project_root(), "data_files", filename)
