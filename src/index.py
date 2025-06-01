import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

def plot_field_ZX_full(
        campo,
        x_range=(-0.10, 0.10),
        z_range=(0, 0.22),
        resolution=1000,
        Rin=0.028, Rex=0.05,
        chamber_length=0.025,
        sigma=3  # Aumenta sigma para mayor suavizado
    ):
    x = campo[:, 0]
    y = campo[:, 1]
    z = campo[:, 2]
    Ex = campo[:, 3]
    Ey = campo[:, 4]
    Ez = campo[:, 5]
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # Malla regular
    z_grid, x_grid = np.mgrid[
        z_range[0]:z_range[1]:complex(resolution),
        x_range[0]:x_range[1]:complex(resolution)
    ]

    points = np.stack([z, x], axis=1)
    E_grid = griddata(
        points, E_mag,
        (z_grid, x_grid),
        method='linear', fill_value=np.nan
    )

    # Suavizado Gaussiano
    E_grid_smooth = gaussian_filter(np.nan_to_num(E_grid, nan=0.0), sigma=sigma)

    # Máscara física: deja NaN SOLO fuera del canal en el chamber
    mask_chamber = (z_grid <= chamber_length) & (
        (np.abs(x_grid) < Rin) | (np.abs(x_grid) > Rex)
    )
    E_grid_masked = E_grid_smooth.copy()
    E_grid_masked[mask_chamber] = np.nan

    # Ajustar rango de color según los valores válidos
    vals_valid = E_grid_masked[~np.isnan(E_grid_masked)]
    vmin = np.nanpercentile(vals_valid, 2) if vals_valid.size > 0 else 1
    vmax = np.nanpercentile(vals_valid, 98) if vals_valid.size > 0 else 10

    norm = mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    cmap = plt.get_cmap('hot').copy()
    cmap.set_under('black')

    fig = plt.figure(figsize=(10, 8))
    plt.gca().set_facecolor('black')

    img = plt.imshow(
        E_grid_masked.T,
        extent=(z_range[0], z_range[1], x_range[0], x_range[1]),
        origin='lower', cmap=cmap, aspect='auto', norm=norm
    )
    cbar = plt.colorbar(img, label='|E| (V/m)')
    ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick:.1e}' for tick in ticks])

    plt.title("Gradiente de la magnitud del campo eléctrico sobre ZX (máscara física en ambos canales)")
    plt.xlabel("Z (m)")
    plt.ylabel("X (m)")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

def plot_field_XY_heatmap_hist(
        campo,
        z_plane=0.025,
        x_range=(-0.08, 0.08),
        y_range=(-0.08, 0.08),
        resolution=400,
        sigma=2.5
    ):
    """
    Visualiza el campo eléctrico |E| en el plano XY a un Z fijo,
    suavizando usando histograma 2D y filtro gaussiano.
    """
    x = campo[:, 0]
    y = campo[:, 1]
    z = campo[:, 2]
    Ex = campo[:, 3]
    Ey = campo[:, 4]
    Ez = campo[:, 5]
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # Seleccionar puntos cercanos al plano Z
    delta_z = (z.max() - z.min()) / 100
    mask_z = np.abs(z - z_plane) < delta_z
    x_data = x[mask_z]
    y_data = y[mask_z]
    E_mag_data = E_mag[mask_z]

    if len(x_data) == 0:
        print("⚠️ No hay puntos en ese plano Z. Prueba con un valor distinto o aumenta delta_z.")
        return

    # Binning: promedio de E_mag en cada celda XY
    H_sum, xedges, yedges = np.histogram2d(
        x_data, y_data, bins=resolution, range=[x_range, y_range], weights=E_mag_data
    )
    H_count, _, _ = np.histogram2d(
        x_data, y_data, bins=resolution, range=[x_range, y_range]
    )
    # Evita división por cero (celdas sin datos)
    H_avg = np.divide(H_sum, H_count, out=np.zeros_like(H_sum), where=H_count > 0)

    # Suavizado gaussiano solo sobre la matriz de promedios
    smooth_grid = gaussian_filter(H_avg, sigma=sigma)

    # Máscara: visualiza solo donde hay datos
    mask = H_count > 0
    smooth_grid_masked = np.where(mask, smooth_grid, np.nan)

    # Escala de colores
    vals_valid = smooth_grid_masked[~np.isnan(smooth_grid_masked)]
    vmin = np.nanpercentile(vals_valid, 2)
    vmax = np.nanpercentile(vals_valid, 98)
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    cmap = plt.get_cmap('inferno').copy()

    plt.figure(figsize=(9, 8))
    plt.gca().set_facecolor('white')

    # Ojo: en histogram2d los ejes van [X, Y] pero imshow espera [Y, X] → Transponer
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    img = plt.imshow(
        smooth_grid_masked.T,  # Transpuesta
        extent=extent,
        origin='lower', cmap=cmap, aspect='auto', norm=norm
    )
    cbar = plt.colorbar(img, label='|E| (V/m)')
    ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick:.1e}' for tick in ticks])

    plt.title(f"Mapa de Calor del Campo Eléctrico |E| en XY, Z={z_plane:.3f} m (bin+suavizado)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.show()

# --- Ejemplo de uso ---
# campo = np.load('data_files/E_Field_Laplace.npy')
# plot_field_XY_heatmap_hist(
#     campo,
#     z_plane=0.025,
#     x_range=(-0.08, 0.08),
#     y_range=(-0.08, 0.08),
#     resolution=400,  # 300-500 es buena densidad para pocos puntos
#     sigma=2.5
# )

def plot_field_ZX_full_with_arrows(
        campo,
        x_range=(-0.10, 0.10),
        z_range=(0, 0.22),
        resolution=1000,
        Rin=0.028, Rex=0.05,
        chamber_length=0.025,
        sigma=3,
        arrow_step=5,           # Submuestreo normal
        arrow_step_channel=30,   # Submuestreo canal
        angulo_max_deg=0.5        # Ángulo máximo para considerar "paralelo a Z"
    ):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import matplotlib.colors as mcolors

    x = campo[:, 0]
    z = campo[:, 2]
    Ex = campo[:, 3]
    Ez = campo[:, 5]
    E_mag = np.sqrt(Ex**2 + Ez**2)

    # Malla regular
    z_grid, x_grid = np.mgrid[
        z_range[0]:z_range[1]:complex(resolution),
        x_range[0]:x_range[1]:complex(resolution)
    ]

    points = np.stack([z, x], axis=1)
    # Interpolación de la magnitud (heatmap)
    E_grid = griddata(
        points, E_mag,
        (z_grid, x_grid),
        method='linear', fill_value=np.nan
    )
    # Interpolación de las componentes vectoriales
    Ex_grid = griddata(points, Ex, (z_grid, x_grid), method='linear', fill_value=np.nan)
    Ez_grid = griddata(points, Ez, (z_grid, x_grid), method='linear', fill_value=np.nan)

    # Suavizado Gaussiano SOLO en la magnitud
    E_grid_smooth = gaussian_filter(np.nan_to_num(E_grid, nan=0.0), sigma=sigma)

    # Máscara física y del canal
    mask_chamber = (z_grid <= chamber_length) & (
        (np.abs(x_grid) < Rin) | (np.abs(x_grid) > Rex)
    )
    mask_channel = (z_grid <= chamber_length) & (np.abs(x_grid) >= Rin) & (np.abs(x_grid) <= Rex)

    E_grid_masked = E_grid_smooth.copy()
    E_grid_masked[mask_chamber] = np.nan
    Ex_grid[mask_chamber] = np.nan
    Ez_grid[mask_chamber] = np.nan

    # Rango de colores del heatmap
    vals_valid = E_grid_masked[~np.isnan(E_grid_masked)]
    vmin = np.nanpercentile(vals_valid, 2) if vals_valid.size > 0 else 1
    vmax = np.nanpercentile(vals_valid, 98) if vals_valid.size > 0 else 10

    norm = mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    cmap = plt.get_cmap('gray').copy()
    cmap.set_under('black')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')

    img = ax.imshow(
        E_grid_masked.T,
        extent=(z_range[0], z_range[1], x_range[0], x_range[1]),
        origin='lower', cmap=cmap, aspect='auto', norm=norm
    )
    cbar = plt.colorbar(img, ax=ax, label='|E| (V/m)')
    ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick:.1e}' for tick in ticks])

    # ---------------------------------------------
    # Flechas negras: SOLO FUERA del canal
    # ---------------------------------------------
    step = arrow_step
    Xq = z_grid[::step, ::step]
    Yq = x_grid[::step, ::step]
    Uq = Ez_grid[::step, ::step]
    Vq = Ex_grid[::step, ::step]

    # Solo donde NO es canal ni mascara
    mask_not_canal = (~mask_channel[::step, ::step]) & (~mask_chamber[::step, ::step]) & (~np.isnan(Uq)) & (~np.isnan(Vq))
    norm_q = np.sqrt(Uq**2 + Vq**2)
    Uq_norm = np.divide(Uq, norm_q, out=np.zeros_like(Uq), where=norm_q!=0)
    Vq_norm = np.divide(Vq, norm_q, out=np.zeros_like(Vq), where=norm_q!=0)

    ax.quiver(
        Xq[mask_not_canal], Yq[mask_not_canal], Uq_norm[mask_not_canal], Vq_norm[mask_not_canal],
        color='black', scale=20, alpha=0.9, width=0.006, zorder=2
    )

    # ---------------------------------------------
    # Flechas magenta: SOLO EN CANAL y casi paralelas a +Z
    # ---------------------------------------------
    step_c = arrow_step_channel
    Xc = z_grid[::step_c, ::step_c]
    Yc = x_grid[::step_c, ::step_c]
    Uc = Ez_grid[::step_c, ::step_c]
    Vc = Ex_grid[::step_c, ::step_c]
    mask_canal = mask_channel[::step_c, ::step_c] & (~np.isnan(Uc)) & (~np.isnan(Vc))

    # Normalización
    norm_c = np.sqrt(Uc**2 + Vc**2)
    Uc_norm = np.divide(Uc, norm_c, out=np.zeros_like(Uc), where=norm_c!=0)
    Vc_norm = np.divide(Vc, norm_c, out=np.zeros_like(Vc), where=norm_c!=0)

    # Flechas en +Z y (más) paralelas al eje Z (más estricto)
    theta = np.abs(np.arctan2(Vc_norm, Uc_norm))   # Ángulo con Z
    angulo_max = np.deg2rad(angulo_max_deg)
    mask_paralelas = (Uc_norm > 0.2) & (theta < angulo_max) & mask_canal
    # (Uc_norm > 0.2) asegura que la componente Z sea suficientemente dominante

    # Chanell
    ax.quiver(
        Xc[mask_paralelas], Yc[mask_paralelas],
        Uc_norm[mask_paralelas], Vc_norm[mask_paralelas],
        color='black', scale=20, alpha=0.95, width=0.005, zorder=3
    )

    ax.set_title("Campo eléctrico sobre ZX (flechas densas y paralelas a +Z en canal)")
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("X (m)")
    fig.tight_layout()
    plt.show()

campo = np.load('data_files/E_Field_Laplace.npy')
plot_field_ZX_full_with_arrows(
        campo,
        x_range=(-0.10, 0.10),
        z_range=(0, 0.22),
        resolution=1000,
        Rin=0.028, Rex=0.05,
        chamber_length=0.025,
        sigma=100,
        arrow_step=70 # Espaciado entre flechas
    )


# plot_field_ZX_full(
#     campo,
#     x_range=(-0.10, 0.10),
#     z_range=(0.0, 0.22),
#     resolution=1000,
#     Rin=0.028,
#     Rex=0.05,
#     chamber_length=0.025,
#     sigma=100  # Prueba valores entre 2 y 6 según la "suavidad" deseada
# )
