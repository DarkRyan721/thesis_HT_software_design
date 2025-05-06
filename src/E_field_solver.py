# --------------------------------------------
# E_field_solver.py
# Class to solve Laplace and Poisson equations using FEniCSx
# Authors: Alfredo Cuellar Valencia, Collin Andrey Sanchez, Miguel Angel Cera
# Purpose: Simulate and visualize electric fields in a Hall effect thruster geometry.
# --------------------------------------------

import os
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
import ufl
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import pyvista as pv
from ufl import grad, inner, dx, SpatialCoordinate

class ElectricFieldSolver:
    """
    ElectricFieldSolver class:
    - Loads a pre-generated mesh (XDMF format)
    - Solves the Laplace or Poisson equations for electric potential
    - Computes the resulting electric field E = -grad(phi)
    - Saves results and optionally visualizes them
    """

    def __init__(self, mesh_file="SimulationZone.xdmf", mesh_folder="data_files"):
        """
        Initializes the solver by loading the mesh and physical tags.
        
        Parameters:
        - mesh_file (str): Path to the mesh XDMF file.
        - mesh_folder (str): Directory where mesh files are located.
        """
        self.mesh_folder = mesh_folder
        self.mesh_file = mesh_file
        self._load_mesh_and_tags()
    
    def _load_mesh_and_tags(self):
        """
        Internal method to load the mesh and associated facet and cell tags
        needed for applying boundary conditions and material properties.
        """
        os.chdir(self.mesh_folder)

        # Load domain (volume mesh)
        with io.XDMFFile(MPI.COMM_WORLD, self.mesh_file, "r") as xdmf:
            self.domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")
        
        # Ensure face-to-volume connectivity is established
        self.domain.topology.create_connectivity(self.domain.topology.dim - 1, self.domain.topology.dim)

        # Load facet (boundary) and cell (domain) tags
        with io.XDMFFile(MPI.COMM_WORLD, self.mesh_file, "r") as xdmf:
            self.facet_tags = xdmf.read_meshtags(self.domain, name="SPT100_Simulation_Zone_facets")
            self.cell_tags = xdmf.read_meshtags(self.domain, name="SPT100_Simulation_Zone_cells")
        
        # Print basic info
        print(f"Mesh loaded: dimension {self.domain.topology.dim}, nodes {self.domain.geometry.x.shape[0]}")
        print(f"Facet tags loaded: {set(self.facet_tags.values)}")

    def _setup_function_space(self):
        """
        Creates the function space (scalar continuous Galerkin) for potential phi.
        """
        return fem.functionspace(self.domain, ("CG", 1))

    def _apply_boundary_conditions(self, V, volt_tag, ground_tag, cathode_tag, Volt, Volt_cath):
        """
        Applies Dirichlet boundary conditions (fixed potentials) on tagged surfaces.
        
        Parameters:
        - V: Function space
        - volt_tag, ground_tag, cathode_tag: Physical tags identifying boundary surfaces
        - Volt: Voltage at the anode
        - Volt_cath: Voltage at the cathode
        """
        u_bc = fem.Function(V)
        u_bc.x.array[:] = np.nan  # Mark all nodes initially as undefined

        boundary_conditions = []

        for tag, value in [(volt_tag, Volt), (ground_tag, 0.0), (cathode_tag, Volt_cath)]:
            dofs = fem.locate_dofs_topological(
                V, self.domain.topology.dim - 1,
                self.facet_tags.indices[self.facet_tags.values == tag]
            )
            u_bc.x.array[dofs] = value
            bc = fem.dirichletbc(u_bc, dofs)
            boundary_conditions.append(bc)

        # Save BC function for visualization
        with io.XDMFFile(self.domain.comm, "boundary_conditions.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(u_bc)

        return boundary_conditions

    def _compute_electric_field(self, phi_h):
        """
        Given the solved potential phi_h, compute the electric field E = -grad(phi).
        
        Parameters:
        - phi_h: Solved potential field
        
        Returns:
        - E_field: Computed electric field (vector field)
        """
        V_vector = fem.functionspace(self.domain, ("CG", 1, (self.domain.geometry.dim,)))
        E_field = fem.Function(V_vector)
        E_expr = fem.Expression(-grad(phi_h), V_vector.element.interpolation_points())
        E_field.interpolate(E_expr)

        # Save the electric field
        with io.XDMFFile(self.domain.comm, "Electric_Field.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(E_field)

        return E_field
    
    def load_density_from_npy(self, path="density_n0.npy"):
        e = -1.602e-19

        rho_array = np.load(path)
        V = fem.functionspace(self.domain, ("CG", 1))
        rho_func = fem.Function(V)
        rho_func.x.array[:] = rho_array*(e / 8.854187817e-12)  # Convertir a rho/epsilon_0
        return rho_func

    def solve_laplace(self, volt_tag=3, ground_tag=6, cathode_tag=7, Volt=300, Volt_cath=18):
        """
        Solves the Laplace equation:
            -div(grad(phi)) = 0
        
        Parameters:
        - volt_tag, ground_tag, cathode_tag: Tags for boundaries
        - Volt: Anode voltage
        - Volt_cath: Cathode voltage
        
        Returns:
        - phi_h: Solved potential field
        - E_field: Computed electric field
        """
        V = self._setup_function_space()
        phi = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = inner(grad(phi), grad(v)) * dx
        L = fem.Constant(self.domain, PETSc.ScalarType(0)) * v * dx

        bcs = self._apply_boundary_conditions(V, volt_tag, ground_tag, cathode_tag, Volt, Volt_cath)

        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        phi_h = problem.solve()

        # Save the potential
        with io.XDMFFile(self.domain.comm, "Laplace.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(phi_h)

        # Calculate E field
        E_field = self._compute_electric_field(phi_h)
        return phi_h, E_field

    def solve_poisson(self, source_term=None, volt_tag=3, ground_tag=6, cathode_tag=7, Volt=300, Volt_cath=18):
        """
        Solves the Poisson equation:
            -div(grad(phi)) = rho/epsilon_0
        
        Parameters:
        - source_term: Right-hand side source term (charge density)
        - volt_tag, ground_tag, cathode_tag: Boundary tags
        - Volt: Anode voltage
        - Volt_cath: Cathode voltage
        
        Returns:
        - phi_h: Solved potential field
        - E_field: Computed electric field
        """
        V = self._setup_function_space()
        phi = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = inner(grad(phi), grad(v)) * dx

        if source_term is None:
            f_expr = fem.Constant(self.domain, PETSc.ScalarType(0))
        else:
            f_expr = source_term
        
        L = f_expr * v * dx

        bcs = self._apply_boundary_conditions(V, volt_tag, ground_tag, cathode_tag, Volt, Volt_cath)

        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        phi_h = problem.solve()

        # Save the potential
        with io.XDMFFile(self.domain.comm, "Poisson.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(phi_h)

        # Calculate E field
        E_field = self._compute_electric_field(phi_h)
        return phi_h, E_field

    def save_electric_field_numpy(self, E_field, filename="Electric_Field_np.npy"):
        """
        Saves the electric field to a .npy file.
        The file contains coordinates and electric field vector components.
        
        Format: [x, y, z, Ex, Ey, Ez]
        """
        X = self.domain.geometry.x
        E_values = E_field.x.array.reshape(-1, self.domain.geometry.dim)
        E_np = np.hstack((X, E_values))
        np.save(filename, E_np)
        print(f"Electric field saved to {filename}")

    def plot_E_Field(self, filename="Electric_Field_np.npy"):
        """
        Visualizes the electric field from a .npy file using PyVista.
        
        Parameters:
        - filename: Path to the saved .npy electric field file.
        """
        E_np = np.load(filename)
        points = E_np[:, :3]
        vectors = E_np[:, 3:]

        magnitudes = np.linalg.norm(vectors, axis=1)
        log_magnitudes = np.log10(magnitudes + 1e-3)

        mesh = pv.PolyData(points)
        mesh["vectors"] = vectors
        mesh["magnitude"] = log_magnitudes

        glyphs = mesh.glyph(orient="vectors", scale=False, factor=0.01)

        plotter = pv.Plotter()
        plotter.set_background("white")
        plotter.add_mesh(glyphs, scalars="magnitude", cmap="plasma")
        plotter.add_axes()
        plotter.add_title("Campo Eléctrico - Dirección y Magnitud")
        plotter.show()


if __name__ == "__main__":
    # Example usage

    solver = ElectricFieldSolver()

    print("\nMesh loaded successfully!")

    # Solve Laplace equation with specific anode voltage
    # Volt_input = 300
    # print("\nSolving Laplace equation...")
    # phi_laplace, E_laplace = solver.solve_laplace(Volt=Volt_input)
    # solver.save_electric_field_numpy(E_laplace, filename="Electric_Field_np.npy")
    # print("Laplace solution completed and saved.")


    # Solve Poisson
    print("\nSolving Poisson equation...")
    source_term= solver.load_density_from_npy()
    phi_poisson, E_poisson = solver.solve_poisson(source_term=source_term)
    solver.save_electric_field_numpy(E_poisson, filename="Electric_Field_np.npy")
    print("Poisson solution completed and saved.")


    # Plot the resulting electric field
    print("\nPlotting the electric field from Laplace solution...")
    solver.plot_E_Field(filename="Electric_Field_np.npy")
    print("\nTest Completed Successfully!")
