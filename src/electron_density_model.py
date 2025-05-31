import numpy as np
from dolfinx import fem, io
import pyvista as pv
from scipy.stats import gamma
from tqdm import tqdm
import os
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from project_paths import data_file


class ElectronDensityModel:
    def __init__(self, domain, r0=0.04, sigma_r=0.007, A=1.2e15, z_min=-0.004, k=2, theta=0.012, plotter = None):
        self.domain = domain
        self.r0 = r0
        self.sigma_r = sigma_r
        self.A = A
        self.z_min = z_min
        self.k = k
        self.theta = theta
        self.density_function = None
        self.plotter = plotter

    def gamma_custom(self, z_val):
        return gamma.pdf(z_val, a=self.k, scale=self.theta, loc=self.z_min)

    def generate_density(self):
        pbar = tqdm(total=4, desc="Densidad de carga")
        V = fem.functionspace(self.domain, ("CG", 1))
        pbar.update(1)

        x_points = V.element.interpolation_points()
        _ = np.zeros(len(x_points))
        pbar.update(1)

        n_e = fem.Function(V)
        n_e.interpolate(lambda x: np.array([
            self.A * np.exp(-((np.sqrt(xi[0]**2 + xi[1]**2) - self.r0)**2) / (2 * self.sigma_r**2)) *
            self.gamma_custom(xi[2]) for xi in x.T]).reshape((1, -1)))

        self.density_function = n_e
        pbar.update(2)
        pbar.close()

        return n_e

    def save_density(self, filename="density_n0.npy"):
        if self.density_function is not None:
            np.save(data_file(filename), self.density_function.x.array)

    @staticmethod
    def plot_density_XY(r0=0.04, sigma_r=0.007, A=1.2e15, z_min=-0.004, k=2, theta=0.012,
                        z_plane=0.01, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1),
                        resolution=5000, Rin=0.028, Rex=0.05):

        x_grid, y_grid = np.mgrid[x_range[0]:x_range[1]:resolution*1j,
                                  y_range[0]:y_range[1]:resolution*1j]
        r_vals = np.sqrt(x_grid**2 + y_grid**2)
        z_vals = np.full_like(x_grid, z_plane)

        density_vals = A * np.exp(-((r_vals - r0)**2) / (2 * sigma_r**2)) * \
                       gamma.pdf(z_vals, a=k, scale=theta, loc=z_min)

        global_max = np.max(density_vals)
        global_min = global_max / 10
        density_vals[density_vals < 1] = 0

        mask = (r_vals >= Rin) & (r_vals <= Rex)
        masked_density = np.ma.masked_where(~mask, density_vals)
        masked_density = np.ma.filled(masked_density, 0)

        fig = plt.figure(figsize=(10, 8))
        fig.canvas.manager.set_window_title("Plano de Densidad Electrónica")
        plt.gca().set_facecolor('black')

        ticks = np.logspace(np.log10(global_min), np.log10(global_max), num=5)
        norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)

        img = plt.imshow(masked_density, extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                         origin='lower', cmap='inferno', aspect='auto', norm=norm)

        cbar = plt.colorbar(img, label='ne (m⁻³)')
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.1e}' for tick in ticks])

        plt.title(f"Densidad de electrones en z = {z_plane}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

    @staticmethod
    def plot_density_ZX(r0=0.04, sigma_r=0.007, A=1.2e15, z_min=-0.004, k=2, theta=0.012,
                        y_plane=0.0, x_range=(0.0, 0.1), z_range=(0, 0.2), resolution=5000,
                        global_max=3.67e16, Rin=0.028, Rex=0.05):

        z_grid, x_grid = np.mgrid[z_range[0]:z_range[1]:resolution*1j,
                                  x_range[0]:x_range[1]:resolution*1j]
        y_vals = np.full_like(x_grid, y_plane)
        r_vals = np.sqrt(x_grid**2 + y_vals**2)

        density_vals = A * np.exp(-((r_vals - r0)**2) / (2 * sigma_r**2)) * \
                       gamma.pdf(z_grid, a=k, scale=theta, loc=z_min)

        global_max = np.max(density_vals)
        global_min = global_max / 1e6
        density_vals[density_vals < 1] = 0

        mask_condition = (z_grid <= 0.02) & ((np.abs(x_grid) < Rin) | (np.abs(x_grid) > Rex))
        masked_density = np.where(mask_condition, 0.0, density_vals)
        masked_density[masked_density < 1] = 0

        fig = plt.figure(figsize=(10, 8))
        fig.canvas.manager.set_window_title("Plano de Densidad Electrónica")
        plt.gca().set_facecolor('black')

        ticks = np.logspace(np.log10(global_min), np.log10(global_max), num=5)
        norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)
        cmap = plt.get_cmap('inferno').copy()
        cmap.set_under('black')

        img = plt.imshow(masked_density.T, extent=(z_range[0], z_range[1], x_range[0], x_range[1]),
                         origin='lower', cmap=cmap, aspect='auto', norm=norm)

        cbar = plt.colorbar(img, label='ne (m⁻³)')
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.1e}' for tick in ticks])

        plt.title(f"Densidad de electrones en Y = {y_plane}")
        plt.xlabel("Z (m)")
        plt.ylabel("X (m)")
        plt.show()


    def plot_density(self, bool_3D=True, bool_XY_Plane=False, bool_XZ_plane=False):
        if bool_3D:
            E_np = np.load(data_file('E_Field_Laplace.npy'))
            points = E_np[:, :3]
            n0 = np.load(data_file('density_n0.npy'))
            n0_log = np.log10(n0)

            mesh = pv.PolyData(points)
            max_value = np.max(n0_log[n0_log != np.log10(1e-100)])
            log_min = max_value - 3.0
            log_max = max_value

            mesh["n0_log"] = n0_log

            if self.plotter is None:
                self.plotter = pv.Plotter()
                use_show = True
            else:
                use_show = False  # Es un QtInteractor

            self.plotter.set_background("white")
            self.plotter.add_axes(color="black")
            try:
                self.plotter.add_mesh(
                    mesh,
                    scalars="n0_log",
                    cmap="plasma",
                    clim=[log_min, log_max],
                    point_size=2,
                    render_points_as_spheres=True,
                    scalar_bar_args={
                        'title': "ne [m⁻³] (log₁₀)\n",
                        'color': 'black',
                        'fmt': "%.1f",
                    }
                )
            except Exception as e:
                print(f"❌ Error al renderizar la malla: {e}")
                return
            # self.plotter.camera_position = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)]
            self.plotter.view_yx()

            self.plotter.add_text("Distribución de Densidad Electrónica", position='upper_edge', font_size=12, color='white')

            if use_show:
                self.plotter.show()
            else:
                self.plotter.reset_camera()
                self.plotter.render()

        elif bool_XY_Plane:
            ElectronDensityModel.plot_density_XY()
        elif bool_XZ_plane:
            ElectronDensityModel.plot_density_ZX()
        else:
            print("Ningún plot seleccionado")


if __name__ == "__main__":
    with io.XDMFFile(MPI.COMM_WORLD, data_file("SimulationZone.xdmf"), "r") as xdmf:
        domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")

    model = ElectronDensityModel(domain)
    # model.generate_density()
    # model.save_density()
    # model.plot_density(bool_3D=True, bool_XY_Plane=False, bool_XZ_plane=False)
    model.plot_density_XY(r0=0.04, sigma_r=0.007, A=1.2e15, z_min=-0.004, k=2, theta=0.012,
                        z_plane=0.01, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1),
                        resolution=5000, Rin=0.028, Rex=0.05)

