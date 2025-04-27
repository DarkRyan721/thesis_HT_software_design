# --------------------------------------------
# fields_solver.py
# Class to solve Laplace and Poisson equations using FEniCSx
# --------------------------------------------

import os
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, mesh
import ufl
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from ufl import grad, inner, dx, exp, SpatialCoordinate
import numpy as np
import pyvista as pv

class ElectricFieldSolver:
    def __init__(self, mesh_file="SimulationZone.xdmf", mesh_folder="data_files"):
        """
        Initializes the solver by loading the mesh and facet tags.
        """
        self.mesh_folder = mesh_folder
        self.mesh_file = mesh_file
        self._load_mesh_and_tags()
    
    def _load_mesh_and_tags(self):
        """
        Loads the mesh and facet tags from the XDMF file.
        """
        # Change to the data_files directory
        os.chdir(self.mesh_folder)

        # Load mesh
        with io.XDMFFile(MPI.COMM_WORLD, self.mesh_file, "r") as xdmf:
            self.domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")
        
        # Create required connectivity
        self.domain.topology.create_connectivity(self.domain.topology.dim - 1, self.domain.topology.dim)

        # Load facet and cell tags
        with io.XDMFFile(MPI.COMM_WORLD, self.mesh_file, "r") as xdmf:
            self.facet_tags = xdmf.read_meshtags(self.domain, name="SPT100_Simulation_Zone_facets")
            self.cell_tags = xdmf.read_meshtags(self.domain, name="SPT100_Simulation_Zone_cells")
        
        # Basic info
        print(f"Mesh loaded: dimension {self.domain.topology.dim}, nodes {self.domain.geometry.x.shape[0]}")
        print(f"Facet tags loaded: {set(self.facet_tags.values)}")

    def _setup_function_space(self):
        """
        Creates a scalar CG(1) function space for the domain.
        """
        return fem.functionspace(self.domain, ("CG", 1))

    def _apply_boundary_conditions(self, V, volt_tag, ground_tag, cathode_tag, Volt, Volt_cath):
        """
        Applies Dirichlet boundary conditions.
        """
        u_bc = fem.Function(V)
        u_bc.x.array[:] = np.nan  # Init as NaN

        boundary_conditions = []

        for tag, value in [(volt_tag, Volt), (ground_tag, 0.0), (cathode_tag, Volt_cath)]:
            dofs = fem.locate_dofs_topological(
                V, self.domain.topology.dim - 1,
                self.facet_tags.indices[self.facet_tags.values == tag]
            )
            u_bc.x.array[dofs] = value
            bc = fem.dirichletbc(u_bc, dofs)
            boundary_conditions.append(bc)

        # Save boundary conditions for visualization
        with io.XDMFFile(self.domain.comm, "boundary_conditions.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(u_bc)

        return boundary_conditions

    def _compute_electric_field(self, phi_h):
        """
        Computes the electric field E = -grad(phi_h)
        """
        V_vector = fem.functionspace(self.domain, ("CG", 1, (self.domain.geometry.dim,)))
        E_field = fem.Function(V_vector)
        E_expr = fem.Expression(-grad(phi_h), V_vector.element.interpolation_points())
        E_field.interpolate(E_expr)

        # Save E field
        with io.XDMFFile(self.domain.comm, "Electric_Field.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(E_field)

        return E_field

    def solve_laplace(self, volt_tag=3, ground_tag=4, cathode_tag=7, Volt=300, Volt_cath=18):
        """
        Solves the Laplace equation: -div(grad(phi)) = 0
        """
        V = self._setup_function_space()
        phi = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = inner(grad(phi), grad(v)) * dx
        L = fem.Constant(self.domain, PETSc.ScalarType(0)) * v * dx

        bcs = self._apply_boundary_conditions(V, volt_tag, ground_tag, cathode_tag, Volt, Volt_cath)

        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        phi_h = problem.solve()

        # Save potential
        with io.XDMFFile(self.domain.comm, "Laplace.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(phi_h)

        # Calculate and return E field
        E_field = self._compute_electric_field(phi_h)
        return phi_h, E_field

    def solve_poisson(self, source_term=None, volt_tag=3, ground_tag=4, cathode_tag=7, Volt=300, Volt_cath=18):
        """
        Solves the Poisson equation: -div(grad(phi)) = rho/epsilon_0
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

        # Save potential
        with io.XDMFFile(self.domain.comm, "Poisson.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(phi_h)

        # Calculate and return E field
        E_field = self._compute_electric_field(phi_h)
        return phi_h, E_field

    def save_electric_field_numpy(self, E_field, filename="Electric_Field_np.npy"):
        """
        Saves the electric field to a .npy file combining node positions and field values.
        """
        X = self.domain.geometry.x
        E_values = E_field.x.array.reshape(-1, self.domain.geometry.dim)
        E_np = np.hstack((X, E_values))
        np.save(filename, E_np)
        print(f"Electric field saved to {filename}")

    def plot_E_Field(self):
        # Cargar datos
        E_np = np.load('Electric_Field_np.npy')
        points = E_np[:, :3]  # X, Y, Z
        vectors = E_np[:, 3:]  # Ex, Ey, Ez

        # Calcular la magnitud para el mapa de calor
        magnitudes = np.linalg.norm(vectors, axis=1)
        log_magnitudes = np.log10(magnitudes + 1e-3)  # Evitamos log(0)

        # Crear un objeto PolyData
        mesh = pv.PolyData(points)

        # Añadir los vectores al objeto
        mesh["vectors"] = vectors
        mesh["magnitude"] = log_magnitudes

        # Crear un glyph (flecha por vector)
        glyphs = mesh.glyph(orient="vectors", scale=False, factor=0.0025)

        # Crear el plotter
        plotter = pv.Plotter()
        plotter.set_background("white")
        plotter.add_mesh(glyphs, scalars="magnitude", cmap="plasma")
        #plotter.add_scalar_bar(title="|E| [V/m]", vertical=True)
        plotter.add_axes()
        plotter.add_title("Campo Eléctrico - Dirección y Magnitud")

        plotter.show()

if __name__ == "__main__":
    E_field = ElectricFieldSolver()

    E_field.plot_E_Field()