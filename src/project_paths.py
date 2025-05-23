import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_files")
WORKER_DIR = os.path.join(PROJECT_ROOT, "workers")

def data_file(filename):
    return os.path.join(DATA_DIR, filename)
def project_file(filename):
    return os.path.join(PROJECT_ROOT, filename)
def temp_data_file(filename):
    return os.path.join(DATA_DIR, filename)
def worker(filename):
    return os.path.join(WORKER_DIR, filename)