# Hall Effect Thruster SPT-100 Simulator

**A High-Fidelity Open-Source Computational Tool for the Analysis and Design of SPT-100 Hall Effect Thrusters**

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![FEniCSx](https://img.shields.io/badge/FEniCSx-2024-blue)](https://fenicsproject.org/)
[![PyVista](https://img.shields.io/badge/PyVista-3D-blue)](https://docs.pyvista.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()

---

## 🛠️ Languages and Tools

<p align="left">
  <a href="https://www.python.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>
  </a>
  <a href="https://www.qt.io/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/0b/Qt_logo_2016.svg" alt="Qt" width="40" height="40"/>
  </a>
  <a href="https://fenicsproject.org/" target="_blank" rel="noreferrer">
    <img src="https://fenicsproject.org/assets/img/fenics-logo-small.png" alt="FEniCSx" width="40" height="40"/>
  </a>
  <a href="https://gmsh.info/" target="_blank" rel="noreferrer">
    <img src="https://gmsh.info/gallery/gmsh_title.png" alt="Gmsh" width="40" height="40"/>
  </a>
  <a href="https://pyvista.org/" target="_blank" rel="noreferrer">
    <img src="https://avatars.githubusercontent.com/u/37482228?s=200&v=4" alt="PyVista" width="40" height="40"/>
  </a>
  <a href="https://numpy.org/" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="NumPy" width="40" height="40"/>
  </a>
  <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg" alt="CUDA" width="40" height="40"/>
  </a>
</p>

---
## Overview

This repository provides an advanced, open-source simulation tool for the analysis and design of **SPT-100 Hall Effect Thrusters**. Developed in Python, the software integrates state-of-the-art scientific libraries such as **FEniCSx** for finite element field solvers, **PyVista** for 3D visualization, and **Gmsh** for automated mesh generation.

The tool features a modular, extensible architecture with a professional GUI (PySide6), empowering both academic researchers and industry engineers to study the complex plasma, electromagnetic, and kinetic phenomena inherent to Hall effect thrusters.

---

## Motivation

The increasing adoption of electric propulsion in both commercial and scientific satellites demands reliable, transparent, and flexible simulation tools. Most existing high-fidelity simulators are either proprietary or highly restricted, limiting access for research groups, universities, and small enterprises.

**This project addresses that gap by providing an open, verifiable, and customizable solution, enabling the exploration and optimization of thruster designs beyond the constraints of closed software.**

---

## Features

- **3D Geometry Generator:** Parametric mesh construction of SPT-100-like thrusters using Gmsh, fully customizable.
- **Electrostatic Field Solver:** Finite Element Method (FEM) solvers for Poisson and Laplace equations (FEniCSx).
- **Magnetic Field Module:** Calculation of solenoidal magnetic fields (Biot-Savart law) with flexible coil geometry.
- **Particle-In-Cell (PIC) Engine:** Kinetic simulation of ion and electron trajectories, supporting full 3D dynamics.
- **Monte Carlo Collisions (MCC):** Stochastic modeling of plasma ionization and neutralization events.
- **Advanced Visualization:** Interactive 3D field, mesh, and particle renderings via PyVista, including field lines, density maps, and trajectories.
- **Graphical User Interface:** Intuitive, multi-panel PySide6-based GUI for configuration, execution, and result analysis.
- **Benchmark Validation:** Comparison with NASA SPT-100 data and other standard reference cases.
- **Open Architecture:** Fully documented, extensible Python code, designed for community contributions.

---

## System Architecture

<!-- If you have an architecture diagram, add it here. Example: -->
<!-- ![System Diagram](docs/system_architecture.png) -->

The workflow consists of the following modular components:

1. **Mesh Generation:** Geometry definition and mesh export (Gmsh, `mesh_generator.py`).
2. **Electrostatic Field Solution:** Laplace/Poisson field calculation (`electron_density_model.py`, FEniCSx).
3. **Magnetic Field Calculation:** Computational analytical magnetic field solver.
4. **Particle Simulation:** 3D PIC-MCC engine for plasma evolution.
5. **Visualization:** 3D post-processing and interactive analysis.
6. **GUI:** User input, monitoring, and batch simulation control.

---

## Scientific Background

This tool implements the Particle-In-Cell with Monte Carlo Collisions (PIC-MCC) method, as described in:

- **Birdsall & Langdon, 2005:** *Plasma Physics via Computer Simulation*
- **NASA SPT-100 Reference Studies**

**Physics Modules Implemented:**

- Poisson and Laplace solvers (electrostatics)
- Biot-Savart and analytical field calculation (magnetics)
- Self-consistent plasma particle simulation
- Stochastic ionization and electron-neutral collisions

Benchmarks and validation were conducted using open data from NASA and leading academic research in Hall effect thrusters.

---

## Standards & Verification

The development and validation processes adhere to established standards:

- IEEE 1620: Software Quality Assurance
- AIAA S-121A: Aerospace Software Standards
- ISO/IEC 12207: Software Lifecycle Processes
- NIST Numerical Methods Guide
- NASA TM-2015-218083: Space Software Verification

---

## Validation & Benchmarks

The simulator has been validated against published SPT-100 test data and compared to reference results in the scientific literature:

- **Geometry & Mesh:** Verified against NASA SPT-100 specifications.
- **Electric Field:** Compared to standard Laplace/Poisson test cases.
- **Plasma Evolution:** Benchmarked with published PIC-MCC results.
- **Performance:** Tested on various hardware (with and without GPU acceleration).

---

## Contributing

Contributions, bug reports, and feature requests are welcome.

- Fork this repo, create a branch, submit a pull request.
- Document new features and tests.
- Adhere to PEP8 and include docstrings.

---

## Authors

- Alfredo Cuellar Valencia
- Collin Andrey Sanchez Diaz
- Miguel Angel Cera

**Supervisors:**

- Dr. Camilo Bayona Roa (Director)
- Dr. David Magin Florez Rubio (Co-Director)
- Dr. Robert Salazar (collaboration and theoretical advisor)

*Pontificia Universidad Javeriana, Bogotá, Colombia*

---

## Installation

> **Requirements:**
> - Python 3.10+
> - FEniCSx (latest)
> - PyVista
> - [Gmsh](https://gmsh.info/)
> - [PySide6](https://pyside.org/)
> - NumPy, SciPy, mpi4py, petsc4py
> - [CuPy](https://cupy.dev/) (for GPU acceleration, requires NVIDIA CUDA Toolkit >= 11.x)
> - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (for GPU acceleration, required by CuPy)


All required Python packages are listed in [`requirements.txt`](./requirements.txt) at the root of this repository.
To review or modify the package list, please refer to that file.

# Install dependencies

```bash
pip install -r requirements.txt

# Clone repository
git clone https://github.com/DarkRyan721/thesis_HT_software_design.git

