# --------------------------------------------
# Gen_Mallado.py
# HallThrusterMesh class - Mesh generator for Hall-effect thruster simulation
# Author: Alfredo Cuellar Valencia, Collin Andrey Sanchez, Miguel Angel Cera
# Purpose: Build a 3D mesh in Gmsh, export it in XDMF format for FEniCSx.
# --------------------------------------------

import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile
import dolfinx.io.gmshio as gmshio
import pyvista as pv
import os
import psutil
from project_paths import data_file


class HallThrusterMesh:
    """
    HallThrusterMesh class:
    - Programmatically builds a 3D geometry based on the SPT-100 thruster.
    - Supports configurable geometry and mesh refinement levels.
    - Generates physical tags for regions and exports mesh for simulation.
    """
    def __init__(self, R_big=0.1/2, R_small=0.056/2, H=0.02, refinement_level="test", min_physical_scale=None, max_elements='auto'):
        """
        Initialize geometric parameters and refinement options.

        Parameters:
        -----------
        R_big : float
            Outer radius of the annular channel. [m]
        R_small : float
            Inner radius of the annular channel. [m]
        H : float
            Axial depth of the thruster. [m]
        refinement_level : str
            Mesh resolution, one of ['low', 'medium', 'high'].
        """
        self.R_big = R_big
        self.R_small = R_small
        self.H = H
        self.max_elements = max_elements
        self.L = 5 * R_big
        self.filename = data_file("SimulationZone")

        self.refinement_level = refinement_level.lower()  # Make sure it's lowercase

        # Define SizeMin and SizeMax based on refinement level

    # def _set_refinement_parameters(self):
    #     """
    #     Internal method to set mesh refinement sizes.
    #     """
    #     if self.refinement_level == "low":
    #         self.size_min = 0.16*self.R_big
    #         self.size_max = 0.24*self.R_big
    #     elif self.refinement_level == "medium":
    #         self.size_min = 0.1*self.R_big
    #         self.size_max = 0.18*self.R_big
    #     elif self.refinemen max_elements = None
    #         raise ValueError("Invalid refinement level. Choose 'low', 'medium', or 'high'.")

        # 1. Cálculo automático o manual del número máximo de elementos
        if max_elements == 'auto' or max_elements is None:
            self.max_elements = self.estimate_max_elements_from_ram()
        else:
            self.max_elements = max_elements

        # 2. Definición del tamaño de la malla según escala física o heurística
        if min_physical_scale is not None:
            self._calculate_mesh_parameters(min_physical_scale, self.refinement_level)
        else:
            self._set_refinement_parameters()

    @staticmethod
    def estimate_max_elements_from_ram():
        """
        Estimate the maximum safe number of mesh elements based on available RAM.
        - Assumes each element consumes approximately 20 KB (including overhead, arrays, and simulation data).
        - Uses 70% of total system RAM as a safety factor.
        - Limits output to between 100,000 and 10,000,000 elements.
        Returns:
            int: Maximum safe number of elements for meshing.
        """
        GB = 1024 ** 3  # Bytes in a gigabyte
        ram_total = psutil.virtual_memory().total / GB  # Total system RAM in GB
        bytes_por_elemento = 20_000  # Estimated bytes per element (20 KB)
        safe_fraction = 0.7  # Use 70% of total RAM to avoid overcommit
        safe_bytes = ram_total * GB * safe_fraction  # Usable bytes
        max_elements = int(safe_bytes // bytes_por_elemento)
        # Limit range to avoid excess or pathological cases
        max_elements = max(100_000, min(max_elements, 10_000_000))
        print(f"[INFO] RAM total: {ram_total:.1f} GB, max_safe_elements estimado: {max_elements}")
        return max_elements

    def _set_refinement_parameters(self):
        """
        Sets mesh size parameters (size_min, size_max) based on refinement level and max_elements.
        - If max_elements is defined, the number of elements is a fraction of max_elements per level:
        - low:   30% of max_elements
        - medium: 60%
        - high:   100%
        - Calculates the target mesh size (h) from domain volume and target_elements.
        - If max_elements is not defined, uses hardcoded heuristics based on the outer radius.
        """
        level_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 1.0
        }
        if self.max_elements is not None:
            if self.refinement_level not in level_map:
                raise ValueError("Invalid refinement level. Choose 'low', 'medium', or 'high'.")
            # Calculate target number of elements
            target_elements = int(self.max_elements * level_map[self.refinement_level])
            Lx = Ly = self.L
            Lz = self.H
            volume = Lx * Ly * Lz
            # Compute mesh cell size to fit within target_elements
            h = (volume / target_elements) ** (1/3)
            self.size_min = h
            self.size_max = 1.5 * h  # Slightly coarser for max
            print(f"[INFO] refinement_level={self.refinement_level}, "
                f"max_elements={self.max_elements}, "
                f"target_elements={target_elements} → h={h:.5f}, "
                f"size_min={self.size_min:.5f}, size_max={self.size_max:.5f}")
        else:
            # Fallback: fixed heuristics (not based on memory)
            if self.refinement_level == "low":
                self.size_min = 0.16 * self.R_big
                self.size_max = 0.24 * self.R_big
            elif self.refinement_level == "medium":
                self.size_min = 0.1 * self.R_big
                self.size_max = 0.18 * self.R_big
            elif self.refinement_level == "high":
                self.size_min = 0.04 * self.R_big
                self.size_max = 0.08 * self.R_big
            else:
                raise ValueError("Invalid refinement level. Choose 'low', 'medium', or 'high'.")
            print(f"[INFO] (heuristic) refinement_level={self.refinement_level} → "
                f"size_min={self.size_min:.5f}, size_max={self.size_max:.5f}")

    def _calculate_mesh_parameters(self, min_physical_scale, refinement_level):
        """
        Adjust mesh size based on a minimum physical scale to be resolved.
        - 'min_physical_scale' defines the smallest physical feature to capture (e.g., Debye length).
        - Applies a factor (2.0, 1.0, 0.5) for 'low', 'medium', 'high' refinement.
        - Ensures total number of elements does not exceed max_safe_elements; if so, h is recalculated.
        - Updates size_min, size_max and self.max_elements accordingly.
        """
        refinement_factors = {
            "low": 2.0,
            "medium": 1.0,
            "high": 0.5
        }
        max_safe_elements = self.max_elements
        if refinement_level not in refinement_factors:
            raise ValueError("Nivel de refinamiento inválido.")
        factor = refinement_factors[refinement_level]
        h = min_physical_scale * factor
        Lx = Ly = self.L
        Lz = self.H
        volume = Lx * Ly * Lz
        n_elements = int(volume / (h ** 3))
        # If too many elements, adjust h to fit within RAM
        if n_elements > max_safe_elements:
            h = (volume / max_safe_elements) ** (1/3)
            n_elements = max_safe_elements
            print(f"[WARN] Se ajusta h a {h:.4g} m para no exceder {max_safe_elements} elementos.")
        self.size_min = h
        self.size_max = 1.5 * h
        self.max_elements = n_elements
        print(f"[INFO] Refinement: {refinement_level}, escala física mínima: {min_physical_scale:.3g} m, "
            f"h usado: {h:.3g} m, elementos estimados: {n_elements}")

    def _set_refinement_parameters(self):
        """
        Internal method to set mesh refinement sizes.
        Now, refinement_level controls what fraction of max_elements you target.
        """
        # Si tienes max_elements definido, usa escalas relativas:
        level_map = {
            "test": 0.001,
            "low": 0.01,
            "medium": 0.04,
            "high": 0.1,
            "ultra": 0.2
        }

        if self.max_elements is not None:
            # Ajusta el número de elementos objetivo según el nivel
            if self.refinement_level not in level_map:
                raise ValueError("Invalid refinement level. Choose 'low', 'medium', or 'high'.")
            target_elements = int(self.max_elements * level_map[self.refinement_level])

            Lx = Ly = self.L
            Lz = self.H
            volume = Lx * Ly * Lz
            h = (volume / target_elements) ** (1/3)
            self.size_min = h
            self.size_max = 1.5 * h
            print(f"[INFO] refinement_level={self.refinement_level}, "
                f"max_elements={self.max_elements}, "
                f"target_elements={target_elements} → h={h:.5f}, "
                f"size_min={self.size_min:.5f}, size_max={self.size_max:.5f}")
        else:
            # Si no se define max_elements, usar los heurísticos antiguos
            if self.refinement_level == "low":
                self.size_min = 0.16 * self.R_big
                self.size_max = 0.24 * self.R_big
            elif self.refinement_level == "medium":
                self.size_min = 0.1 * self.R_big
                self.size_max = 0.18 * self.R_big
            elif self.refinement_level == "high":
                self.size_min = 0.04 * self.R_big
                self.size_max = 0.08 * self.R_big
            else:
                raise ValueError("Invalid refinement level. Choose 'low', 'medium', or 'high'.")
            print(f"[INFO] (heuristic) refinement_level={self.refinement_level} → size_min={self.size_min:.5f}, size_max={self.size_max:.5f}")

    def create_mesh(self, comm: MPI.Comm, model: gmsh.model, name: str):
        """
        Convert the Gmsh model into a dolfinx mesh and export it to an XDMF file.
        """
        msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0)
        msh.name = name
        ct.name = f"{msh.name}_cells"
        ft.name = f"{msh.name}_facets"
        msh.topology.create_connectivity(2, 3)

        with XDMFFile(msh.comm, self.filename + ".xdmf", "w") as file:
            file.write_mesh(msh)
            file.write_meshtags(ct, msh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
            file.write_meshtags(ft, msh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")

    def find_matching_surfaces(self, reference_coords, all_surfaces, tol=1e-6):
        """
        Find surfaces whose center of mass matches reference coordinates (within tolerance).
        """
        matched = []
        for ref in reference_coords:
            for dim, tag in all_surfaces:
                com = gmsh.model.occ.getCenterOfMass(dim, tag)
                if np.linalg.norm(np.array(com) - np.array(ref)) < tol:
                    matched.append(tag)
        return matched

    def visualize_mesh_views(mesh_path,show_grid=True):
        """
        Visualize the mesh using PyVista in XY, YZ, XZ planes and isometric view.

        Parameters:
            mesh_path (str): Path to the .msh file exported from Gmsh
        """

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        print(f"Loading mesh from {mesh_path} ...")
        mesh = pv.read(mesh_path)

        yz_slice = mesh.slice(normal='x')  # YZ plane

        mesh_color = "lightblue"
        edge_color = "black"
        bg_color = "white"

        # -------------------
        # YZ Plane View
        # -------------------
        plotter_yz = pv.Plotter(window_size=(800, 600))
        plotter_yz.set_background(bg_color)
        plotter_yz.add_text("Plano YZ", position="upper_edge", font_size=14)
        plotter_yz.add_mesh(yz_slice, color=mesh_color, show_edges=True, edge_color=edge_color, line_width=1)
        plotter_yz.view_yz()
        if show_grid:
            plotter_yz.show_axes()
            plotter_yz.show_bounds(
                xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
                location='outer', all_edges=True, grid='front'
            )
        plotter_yz.show()

        # -------------------
        # Isometric View
        # -------------------
        plotter_iso = pv.Plotter(window_size=(800, 600))
        plotter_iso.set_background(bg_color)
        plotter_iso.add_text("Vista Isométrica", position="upper_edge", font_size=14)
        plotter_iso.add_mesh(mesh, color=mesh_color, opacity=0.5, show_edges=True, edge_color=edge_color, line_width=0.8)
        plotter_iso.view_isometric()
        if show_grid:
            plotter_iso.show_axes()
            plotter_iso.show_bounds(
                xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
                location='outer', all_edges=True, grid='front'
            )
        plotter_iso.show()

    def generate(self):
        """
        Main method to build geometry, assign physical groups, apply mesh refinement,
        generate the mesh, and export it.
        """
        gmsh.initialize()
        gmsh.model.add("SPT100_Simulation_Zone")
        R_big, R_small, H, L = self.R_big, self.R_small, self.H, self.L

        pos_cube = 1.5 * (L / 2)

        # ----------------------------------------------------------------------
        # Geometry Construction
        #(Don't alter the order of the code, because there could be changes in IDs that are used
        #afterwards)
        # ----------------------------------------------------------------------
        #Hollow Cathode Points
        point_1 = gmsh.model.occ.add_point(0.4*R_big,1.2*R_big,H)
        point_2 = gmsh.model.occ.add_point(-0.4*R_big,1.2*R_big,H)
        point_3 = gmsh.model.occ.add_point(0.4*R_big,1.8*R_big,H)
        point_4 = gmsh.model.occ.add_point(-0.4*R_big,1.8*R_big,H)
        point_5 = gmsh.model.occ.add_point(0.4*R_big,1.5*R_big,H+0.015)
        point_6 = gmsh.model.occ.add_point(-0.4*R_big,1.5*R_big,H+0.015)

        #Plume domain points
        point_7 = gmsh.model.occ.add_point(pos_cube,pos_cube,H)
        point_8 = gmsh.model.occ.add_point(pos_cube,-pos_cube,H)
        point_9 = gmsh.model.occ.add_point(-pos_cube,pos_cube,H)
        point_10 = gmsh.model.occ.add_point(-pos_cube,-pos_cube,H)
        point_11 = gmsh.model.occ.add_point(pos_cube,pos_cube,H+0.2)
        point_12 = gmsh.model.occ.add_point(pos_cube,-pos_cube,H+0.2)
        point_13 = gmsh.model.occ.add_point(-pos_cube,pos_cube,H+0.2)
        point_14 = gmsh.model.occ.add_point(-pos_cube,-pos_cube,H+0.2)

        #Hollow Cathode lines
        l1 = gmsh.model.occ.add_line(point_1,point_2)
        l2 = gmsh.model.occ.add_line(point_3,point_4)
        l3 = gmsh.model.occ.add_line(point_1,point_3)
        l4 = gmsh.model.occ.add_line(point_2,point_4)
        l5 = gmsh.model.occ.add_line(point_2,point_6)
        l6 = gmsh.model.occ.add_line(point_4,point_6)
        l7 = gmsh.model.occ.add_line(point_1,point_5)
        l8 = gmsh.model.occ.add_line(point_3,point_5)
        l9 = gmsh.model.occ.add_line(point_5,point_6)

        #Plume domain Lines
        l10 = gmsh.model.occ.add_line(point_7,point_8)
        l11 = gmsh.model.occ.add_line(point_8,point_10)
        l12 = gmsh.model.occ.add_line(point_10,point_9)
        l13 = gmsh.model.occ.add_line(point_9,point_7)
        l14 = gmsh.model.occ.add_line(point_11,point_12)
        l15 = gmsh.model.occ.add_line(point_12,point_14)
        l16 = gmsh.model.occ.add_line(point_14,point_13)
        l17 = gmsh.model.occ.add_line(point_13,point_11)

        l18 = gmsh.model.occ.add_line(point_9,point_13)
        l19 = gmsh.model.occ.add_line(point_7,point_11)
        l20 = gmsh.model.occ.add_line(point_8,point_12)
        l21 = gmsh.model.occ.add_line(point_10,point_14)

        l22 = gmsh.model.occ.add_line(point_9,point_4)
        l23 = gmsh.model.occ.add_line(point_7,point_3)
        l24 = gmsh.model.occ.add_line(point_10,point_2)
        l25 = gmsh.model.occ.add_line(point_8,point_1)

        #Hollow Cathode Curve loops
        curve_loop_hollow1 = gmsh.model.occ.add_curve_loop([l4,l5,l6])
        curve_loop_hollow2 = gmsh.model.occ.add_curve_loop([l3,l7,l8])
        curve_loop_hollow3 = gmsh.model.occ.add_curve_loop([l2,l8,l6,l9])
        curve_loop_hollow4 = gmsh.model.occ.add_curve_loop([l9,l5,l1,l7])

        #Plume Domain Curve Loops
        curve_loop_plume1 = gmsh.model.occ.add_curve_loop([l22,l2,l23,l13])
        curve_loop_plume2 = gmsh.model.occ.add_curve_loop([l17,l14,l15,l16])
        curve_loop_plume3 = gmsh.model.occ.add_curve_loop([l13,l19,l17,l18])
        curve_loop_plume4 = gmsh.model.occ.add_curve_loop([l10,l19,l14,l20])
        curve_loop_plume5 = gmsh.model.occ.add_curve_loop([l11,l21,l15,l20])
        curve_loop_plume6 = gmsh.model.occ.add_curve_loop([l21,l12,l18,l16])
        curve_loop_plume7 = gmsh.model.occ.add_curve_loop([l23,l3,l25,l10])
        curve_loop_plume8 = gmsh.model.occ.add_curve_loop([l24,l1,l25,l11])
        curve_loop_plume9 = gmsh.model.occ.add_curve_loop([l24,l4,l22,l12])

        #Surface Construction
        surface_plume1 = gmsh.model.occ.add_plane_surface([curve_loop_plume1])
        surface_plume2 = gmsh.model.occ.add_plane_surface([curve_loop_plume2])
        surface_plume3 = gmsh.model.occ.add_plane_surface([curve_loop_plume3])
        surface_plume4 = gmsh.model.occ.add_plane_surface([curve_loop_plume4])
        surface_plume5 = gmsh.model.occ.add_plane_surface([curve_loop_plume5])
        surface_plume6 = gmsh.model.occ.add_plane_surface([curve_loop_plume6])
        surface_plume7 = gmsh.model.occ.add_plane_surface([curve_loop_plume7])
        surface_plume8 = gmsh.model.occ.add_plane_surface([curve_loop_plume8])
        surface_plume9 = gmsh.model.occ.add_plane_surface([curve_loop_plume9])
        surface_hollow1 = gmsh.model.occ.add_plane_surface([curve_loop_hollow1])
        surface_hollow2 = gmsh.model.occ.add_plane_surface([curve_loop_hollow2])
        surface_hollow3 = gmsh.model.occ.add_plane_surface([curve_loop_hollow3])
        surface_hollow4 = gmsh.model.occ.add_plane_surface([curve_loop_hollow4])

        #Saving original COM of the surfaces for easily arragind physical groups
        cathode_surfaces = [surface_hollow3,surface_hollow4,surface_hollow1,surface_hollow2]
        
        outlet_plume_surfaces = [surface_plume1,surface_plume2,surface_plume3,surface_plume4
                                ,surface_plume5,surface_plume6,surface_plume7,surface_plume8
                                ,surface_plume9]
        
        original_cathode_coords = [
            gmsh.model.occ.getCenterOfMass(2,s) for s in cathode_surfaces
        ]

        original_outlet_coords = [
            gmsh.model.occ.getCenterOfMass(2,s) for s in outlet_plume_surfaces
        ]
        surface_loop_cathode_plume = gmsh.model.occ.add_surface_loop([surface_plume1,
        surface_plume2,surface_plume3,surface_plume4,surface_plume5,surface_plume6,
        surface_hollow1,surface_hollow2,surface_hollow3,surface_hollow4,
        surface_plume7,surface_plume8,surface_plume9])

        vol_cathode_plume = gmsh.model.occ.add_volume([surface_loop_cathode_plume])

        gmsh.model.occ.synchronize()

        # Cylindrical channel body: outer and inner cylinders
        cylinder_outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, R_big)
        cylinder_inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, R_small)

        # Subtract inner cylinder from outer to create a hollow ring
        cylinder = gmsh.model.occ.cut([(3, cylinder_outer)], [(3, cylinder_inner)])
        gmsh.model.occ.synchronize()

        # Fragment geometry to ensure intersection and meshing coherence
        fragmented = gmsh.model.occ.fragment([(3, vol_cathode_plume)], cylinder[0])
        gmsh.model.occ.synchronize()

        # ----------------------------------------------------------------------
        # Assign physical groups (volumes and surfaces)
        # ----------------------------------------------------------------------
        vol_cathode_plume = 1
        cylinder_volume = 2

        gmsh.model.addPhysicalGroup(3, [vol_cathode_plume], 1)
        gmsh.model.setPhysicalName(3, 1, "Plume_Domain")
        gmsh.model.addPhysicalGroup(3, [cylinder_volume], 2)
        gmsh.model.setPhysicalName(3, 2, "Thruster_Domain")


        # Recover surfaces after geometry operations
        all_surfaces = gmsh.model.getEntities(2)
        outlet_plume_surfaces_new = self.find_matching_surfaces(original_outlet_coords, all_surfaces)
        cathode_surfaces_new = self.find_matching_surfaces(original_cathode_coords,all_surfaces)
        cathode_surfaces_new.remove(29)
        cathode_surfaces_new.remove(28)
        cathode_surfaces_new.remove(27)

        # Identify inlet, outlet, and wall surfaces by their center-of-mass coordinates
        inlet_surfaces = []
        outlet_thruster_surfaces = []
        cylinder_wall_surfaces = []

        tol = 1e-3
        for s in all_surfaces:
            com = gmsh.model.occ.getCenterOfMass(s[0],s[1])
            if np.isclose(com[2],0,atol=tol):
                inlet_surfaces.append(s[1])

        outlet_thruster_surfaces = [19]
        cylinder_wall_surfaces = [17,18]
        outlet_plume_surfaces_new.append(32)
        outlet_plume_surfaces_new.append(33)
        outlet_plume_surfaces_new.append(27)
        outlet_plume_surfaces_new.append(28)
        outlet_plume_surfaces_new.append(29)

        # Assign surface physical groups with appropriate tags
        gmsh.model.addPhysicalGroup(2,cathode_surfaces_new,7)
        gmsh.model.setPhysicalName(2, 7, "Cathode_walls")

        gmsh.model.addPhysicalGroup(2, inlet_surfaces, 3)
        gmsh.model.setPhysicalName(2, 3, "Gas_inlet")

        gmsh.model.addPhysicalGroup(2, outlet_thruster_surfaces, 4)
        gmsh.model.setPhysicalName(2, 4, "Thruster_outlet")

        gmsh.model.addPhysicalGroup(2, cylinder_wall_surfaces, 5)
        gmsh.model.setPhysicalName(2, 5, "Walls")

        gmsh.model.addPhysicalGroup(2, outlet_plume_surfaces_new, 6)
        gmsh.model.setPhysicalName(2, 6, "Plume_outlet")

        # ----------------------------------------------------------------------
        # Local Mesh concentration (NEW: uses dynamic self.size_min and self.size_max)
        # ----------------------------------------------------------------------
        concentration_surfaces = cathode_surfaces_new + inlet_surfaces + cylinder_wall_surfaces + outlet_thruster_surfaces

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "FacesList", concentration_surfaces)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", self.size_min)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", self.size_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.02*R_big)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.16*R_big)

        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        # ----------------------------------------------------------------------
        # Final Mesh generation and Export
        # ----------------------------------------------------------------------
        gmsh.model.mesh.generate(3)
        #gmsh.fltk.run()
        gmsh.write(self.filename + ".msh")
        self.create_mesh(MPI.COMM_WORLD, gmsh.model, name="SPT100_Simulation_Zone")
        gmsh.finalize()

if __name__ == "__main__":
    # From GUI inputs:
    outer_radius = 0.1/2 #0.1/2
    inner_radius = 0.056/2 #0.056/2
    height = 0.02 #0.02
    refinement = "test"

    mesh_gen = HallThrusterMesh(
        R_big=outer_radius,
        R_small=inner_radius,
        H=height,
        refinement_level=refinement,
        max_elements='auto'  # automático según RAM
    )
    mesh_gen.generate()
    HallThrusterMesh.visualize_mesh_views(data_file("SimulationZone.msh"), show_grid=True)
