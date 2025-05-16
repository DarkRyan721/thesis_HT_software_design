from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
from numpy import sin, cos, pi


#_____________________________________________________________________________________________________________
#                   1] Parametros iniciales del campo magnetico

# Numero de partes del solenoide
nSteps = 1500

# Longitud, Radio y # vueltas del solenoide
L = 0.021 #[m]
Rin = 0.027 #[m]
Rout = 0.8*Rin
Rext = 0.05
N = 150  #[1]

# Constante de permiabilidad del vacio y corriente del solenoides
muo = (4e-7)*np.pi #[(T*m)/A]
i = 15 #[A]

#_____________________________________________________________________________________________________________
#                   2] Funcion que calcula la magnitud del campo en un punto (X,Y,Z)
def Magnetic_Field(x_grid, y_grid, z_grid, X_sol, Y_sol, Z_sol, chunk_size=100):
    # Asegurar que las salidas tengan la misma forma que la malla de entrada
    xg = np.array(x_grid, dtype=np.float64)
    yg = np.array(y_grid, dtype=np.float64)
    zg = np.array(z_grid, dtype=np.float64)

    Xs = np.array(X_sol, dtype=np.float64)
    Ys = np.array(Y_sol, dtype=np.float64)
    Zs = np.array(Z_sol, dtype=np.float64)

    # Inicializar acumuladores en la GPU
    Bx = np.zeros_like(xg, dtype=np.float64)
    By = np.zeros_like(yg, dtype=np.float64)
    Bz = np.zeros_like(zg, dtype=np.float64)

    # Segmentos diferenciales del solenoide
    dlx = Xs[1:] - Xs[:-1]
    dly = Ys[1:] - Ys[:-1]
    dlz = Zs[1:] - Zs[:-1]

    # 2) Bucle en pasos (o chunks)
    #    Usamos range(0, nSteps-1, chunk_size) para partir en trozos
    for start in tqdm(range(0, nSteps - 1, chunk_size),
                    desc="Calculando campo magnético (GPU)",
                    unit="chunk"):
        end = min(start + chunk_size, nSteps - 1)

        # Extraemos el trozo de las dl y posiciones
        dlx_chunk = dlx[start:end]
        dly_chunk = dly[start:end]
        dlz_chunk = dlz[start:end]

        X_sol_chunk = Xs[start:end]
        Y_sol_chunk = Ys[start:end]
        Z_sol_chunk = Zs[start:end]

        # Ahora sumamos la contribución de cada uno de esos segmentos
        for k in range(len(dlx_chunk)):
            rx = xg - X_sol_chunk[k]
            ry = yg - Y_sol_chunk[k]
            rz = zg - Z_sol_chunk[k]

            rnorm = np.sqrt(rx*rx + ry*ry + rz*rz)
            # Evitar dividir por cero
            rnorm = np.maximum(rnorm, 1e-14)

            # dl x r
            dBx = (dly_chunk[k] * rz - dlz_chunk[k] * ry) / (rnorm**3)
            dBy = (dlz_chunk[k] * rx - dlx_chunk[k] * rz) / (rnorm**3)
            dBz = (dlx_chunk[k] * ry - dly_chunk[k] * rx) / (rnorm**3)

            # Acumular al campo total
            Bx += dBx
            By += dBy
            Bz += dBz

        # (Opcional) Sincronizar el GPU para que tqdm no avance antes de tiempo
        # np.cuda.Stream.null.synchronize()

    # 3) Aplicar la constante de Biot-Savart
    factor = (muo * i) / (4.0 * np.pi)
    Bx *= factor
    By *= factor
    Bz *= factor

    # Magnitud
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # 4) Devolver a CPU
    # return Bx.get(), By.get(), Bz.get(), B_mag.get()
    return Bx, By, Bz, B_mag

#_____________________________________________________________________________________________________________
#                       3] Creacion de puntos del solenoide


phi = np.linspace(0, 2*np.pi, nSteps)
def xParametrique_inner( phi ) : return Rin*cos(N*phi)
def yParametrique_inner( phi ) : return Rin*sin(N*phi)
def zParametrique_inner( phi ) : return L*phi/(2*pi)

def xParametrique_outer( phi ) : return Rout*np.cos(N*phi)
def yParametrique_outer( phi ) : return Rout*np.sin(N*phi)
def zParametrique_outer( phi ) : return L*phi/(2*np.pi)

X_inner = xParametrique_inner(phi)
Y_inner = yParametrique_inner(phi)
Z_inner = zParametrique_inner(phi)

X_temp = Rext*np.cos(N*phi)
Y_temp = Rext*np.sin(N*phi)
Z_temp = L*phi/(2*np.pi)

X1 = xParametrique_outer(phi) + Rext + Rout/2
Y1 = yParametrique_outer(phi) + Rext + Rout/2
Z1 = zParametrique_outer(phi)

X2 = xParametrique_outer(phi) - Rext - Rout/2
Y2 = yParametrique_outer(phi) + Rext + Rout/2
Z2 = zParametrique_outer(phi)

X3 = xParametrique_outer(phi) - Rext - Rout/2
Y3 = yParametrique_outer(phi) - Rext - Rout/2
Z3 = zParametrique_outer(phi)

X4 = xParametrique_outer(phi) + Rext + Rout/2
Y4 = yParametrique_outer(phi) - Rext - Rout/2
Z4 = zParametrique_outer(phi)

#_____________________________________________________________________________________________________________
#                       4] Funcion de ploteo de los cables de los solenoides

def wire_plot():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot3D(X_inner, Y_inner, Z_inner, 'black')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect('equal')

    plt.show()

#_____________________________________________________________________________________________________________
#                       4] Funcion para ploteo de los solenoides

def Solenoid_points_plot():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # ax.plot3D(X, Y, Z, 'black')
    ax.scatter(X_inner, Y_inner, Z_inner, "black")
    ax.scatter(X_temp, Y_temp, Z_temp, "black")
    ax.scatter(X1, Y1, Z1, 'black')
    ax.scatter(X2, Y2, Z2, 'black')
    ax.scatter(X3, Y3, Z3, 'black')
    ax.scatter(X4, Y4, Z4, 'black')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect('equal')

    plt.show()

#_____________________________________________________________________________________________________________
#                       5] Funcion para vista lateral del campo magnetico

def color_map_B():
    # Generar la malla de evaluación
    yVal_focus = np.linspace(-0.150, 0.150, 100)  # Reducir el rango en Y
    zVal_focus = np.linspace(-0.060, 0.060, 100)  # Mantener Z igual

    # Crear la malla de puntos
    YVec_focus, ZVec_focus = np.meshgrid(yVal_focus, zVal_focus)

    # Calcular la magnitud del campo en la malla
    _,_,_,B_inner = Magnetic_Field(0.001 * np.ones_like(YVec_focus), YVec_focus, ZVec_focus, X_inner, Y_inner, Z_inner)
    _,_,_,B1 = Magnetic_Field(0.001 * np.ones_like(YVec_focus), YVec_focus, ZVec_focus, X1, Y1, Z1)
    _,_,_,B2 = Magnetic_Field(0.001 * np.ones_like(YVec_focus), YVec_focus, ZVec_focus, X2, Y2, Z2)
    _,_,_,B3 = Magnetic_Field(0.001 * np.ones_like(YVec_focus), YVec_focus, ZVec_focus, X3, Y3, Z3)
    _,_,_,B4 = Magnetic_Field(0.001 * np.ones_like(YVec_focus), YVec_focus, ZVec_focus, X4, Y4, Z4)

    B_values = np.abs(B_inner - (B2+B1+B3+B4))

    num_levels = 10
    contour_levels = np.linspace(np.min(B_values[B_values > 0]), np.max(B_values), num_levels)

    # Crear la visualización con escala logarítmica para mejorar el contraste
    fig, ax = plt.subplots(figsize=(10, 8))

    # Usar escala logarítmica para resaltar detalles en valores pequeños
    im = ax.imshow(B_values, cmap='plasma', norm=colors.LogNorm(vmin=np.max([B_values.min(), 1e-6]), vmax=B_values.max()),
                extent=[yVal_focus.min(), yVal_focus.max(), zVal_focus.min(), zVal_focus.max()], origin='lower')

    # Ajustar niveles de contorno
    num_levels = 30
    contour_levels = np.logspace(np.log10(np.max([B_values.min(), 1e-6])), np.log10(B_values.max()), num_levels)

    # Agregar líneas de contorno con escala logarítmica
    cset = ax.contour(YVec_focus, ZVec_focus, B_values, levels=contour_levels, linewidths=1, colors='white', norm=colors.LogNorm())
    ax.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

    # Agregar barra de color con escala logarítmica
    cbar = plt.colorbar(im, ax=ax, label="|B| (Escala Log)")

    # Etiquetas y título
    ax.set_xlabel("Y-axis")
    ax.set_ylabel("Z-axis")
    ax.set_title("Magnitud del Campo Magnético (Escala Log)")

    ax.set_aspect('auto')

    # Mostrar la figura
    plt.show()

#_____________________________________________________________________________________________________________
#                       6] Funcion para observar las lineas de campo magnetico

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import matplotlib.colors as colors
from numpy import sin, cos, pi


def Field_view():
    x_grid = np.linspace(-0.08, 0.08, 40)
    y_grid = np.linspace(-0.08, 0.08, 40)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)

    # Calcular el campo
    Bx_inner, By_inner, _, _ = Magnetic_Field(X, Y, Z, X_inner, Y_inner, Z_inner)
    Bx1, By1, _, _ = Magnetic_Field(X, Y, Z, X1, Y1, Z1)
    Bx2, By2, _, _ = Magnetic_Field(X, Y, Z, X2, Y2, Z2)
    Bx3, By3, _, _ = Magnetic_Field(X, Y, Z, X3, Y3, Z3)
    Bx4, By4, _, _ = Magnetic_Field(X, Y, Z, X4, Y4, Z4)

    Bx = Bx_inner - (Bx1 + Bx2 + Bx3 + Bx4)
    By = By_inner - (By1 + By2 + By3 + By4)

    max_arrow = 0.003
    Bx_l = np.clip(Bx, -max_arrow, max_arrow)
    By_l = np.clip(By, -max_arrow, max_arrow)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(X, Y, Bx_l, By_l,
              angles='xy', scale_units='xy', scale=1, color='b')

    # Añadir círculos que representan la proyección de los cilindros
    circ1 = patches.Circle((0, 0), Rin,
                           fill=False,
                           edgecolor='gray',
                           linestyle='--',
                           alpha=0.5,
                           linewidth=2,
                           label=f'Rin = {Rin} m')
    circ2 = patches.Circle((0, 0), Rout,
                           fill=False,
                           edgecolor='black',
                           linestyle=':',
                           alpha=0.5,
                           linewidth=2,
                           label=f'Rout = {Rout:.4f} m')
    ax.add_patch(circ1)
    ax.add_patch(circ2)

    # Ajustes finales
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Líneas del Campo Magnético en el Plano XY\ncon cilindros Rin y Rout')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right')
    plt.show()


import numpy as np
import pyvista as pv


def solenoid_pyvista_plot():
    phi = np.linspace(0, 2 * np.pi, nSteps)

    def inner_pts():
        return np.column_stack((Rin * np.cos(N*phi),
                                Rin * np.sin(N*phi),
                                L   * phi/(2*np.pi)))
    def outer_pts(offset_x, offset_y):
        return np.column_stack(( (0.8*Rin) * np.cos(N*phi) + offset_x,
                                 (0.8*Rin) * np.sin(N*phi) + offset_y,
                                 L   * phi/(2*np.pi)))
    def thin_pts():
        return np.column_stack((Rext * np.cos(N*phi),
                                Rext * np.sin(N*phi),
                                L    * phi/(2*np.pi)))

    p = pv.Plotter(window_size=(900, 600))

    # Dibujamos las espiras como antes
    bundles = [
        (inner_pts(),   {'color': 'red',    'radius': 0.0005}),
        (thin_pts(),    {'color': 'orange', 'radius': 0.0003}),
        (outer_pts( Rext+Rout/2,  Rext+Rout/2), {'color': 'blue',   'radius': 0.0005}),
        (outer_pts(-Rext-Rout/2,  Rext+Rout/2), {'color': 'green',  'radius': 0.0005}),
        (outer_pts(-Rext-Rout/2, -Rext-Rout/2), {'color': 'purple', 'radius': 0.0005}),
        (outer_pts( Rext+Rout/2, -Rext-Rout/2), {'color': 'cyan',   'radius': 0.0005}),
    ]
    for pts, opts in bundles:
        nPts  = pts.shape[0]
        cells = np.hstack([[nPts], np.arange(nPts)])
        line  = pv.PolyData(pts, lines=cells)
        tube  = line.tube(radius=opts['radius'], n_sides=16)
        p.add_mesh(tube, color=opts['color'], smooth_shading=True)

    # 1) Cilindro interior: superficie semitransparente
    cyl_in = pv.Cylinder(center=(0,0,L/2), direction=(0,0,1),
                         radius=Rin, height=L)
    p.add_mesh(cyl_in,
               color='gray',
               opacity=0.2,
               smooth_shading=False,
               style='surface')

    # 2) Cilindro exterior: solo armazón (wireframe) para destacar contorno
    cyl_out = pv.Cylinder(center=(0,0,L/2), direction=(0,0,1),
                          radius=Rout, height=L)
    p.add_mesh(cyl_out,
               color='black',
               opacity=0.3,
               smooth_shading=False,
               style='wireframe',
               line_width=2)

    # Ajustes de escena
    p.add_axes(line_width=2)
    p.enable_eye_dome_lighting()
    p.set_background("white")
    p.camera_position = 'iso'

    p.show()

def save_field(filename_prefix, Xg, Yg, Zg, Bx, By, Bz, Bmag, format='npz'):
    """
    Guarda los arrays del campo magnético en disco.

    Parámetros:
    - filename_prefix: ruta+nombre base (sin extensión).
    - Xg, Yg, Zg: mallas de coordenadas (shape: [nx,ny,nz]).
    - Bx, By, Bz: componentes del campo.
    - Bmag: magnitud del campo.
    - format: 'npy' o 'npz'.
    
    Diferencias:
    - .npy almacena **un solo** array (usando np.save).
    - .npz es un contenedor ZIP con **varios** arrays (usando np.savez[ _compressed]).
    """
    print(f"Guardando campo magnético en {filename_prefix}...")
    if format == 'npz':
        # empaqueta todos los arrays en un sólo .npz
        np.savez_compressed(
            filename_prefix + '.npz',
            X=Xg, Y=Yg, Z=Zg,
            Bx=Bx, By=By, Bz=Bz, Bmag=Bmag
        )
        print(f"Guardado en {filename_prefix}.npz")
    elif format == 'npy':
        # guarda cada array por separado
        np.save(filename_prefix + '_X.npy', Xg)
        np.save(filename_prefix + '_Y.npy', Yg)
        np.save(filename_prefix + '_Z.npy', Zg)
        np.save(filename_prefix + '_Bx.npy', Bx)
        np.save(filename_prefix + '_By.npy', By)
        np.save(filename_prefix + '_Bz.npy', Bz)
        np.save(filename_prefix + '_Bmag.npy', Bmag)
        print(f"Guardado en {filename_prefix}_*.npy")
    else:
        raise ValueError("format debe ser 'npy' o 'npz'")

def plot_field_pyvista(Xg, Yg, Zg, Bx, By, Bz, Bmag,
                       arrow_factor=0.5, arrow_spacing=4):
    import pyvista as pv, numpy as np

    grid = pv.StructuredGrid(Xg, Yg, Zg)
    vectors = np.stack([Bx.ravel(), By.ravel(), Bz.ravel()], axis=1)
    grid.point_data['Bvec'] = vectors
    grid.point_data['Bmag'] = Bmag.ravel()

    # 1) Cortamos el slice en z = L/2
    slice_plane = grid.slice(normal='z', origin=(0,0,L/2))

    p = pv.Plotter()
    # 2) Dibujamos el slice coloreado por Bmag
    p.add_mesh(slice_plane, scalars='Bmag', cmap='viridis', show_scalar_bar=True)

    # 3) Glyphs sobre el slice
    glyphs = slice_plane.glyph(orient='Bvec', scale='Bmag', factor=arrow_factor,
                               tolerance=0.01, geom=pv.Arrow())
    p.add_mesh(glyphs, color='black')

    p.add_axes()
    p.show()


# # o guardar múltiples .npy
# # 
# # 3D: malla 30×30×15
# x = np.linspace(-0.1, 0.1, 30)
# y = np.linspace(-0.1, 0.1, 30)
# z = np.linspace( 0.0, 0.05, 15)
# Xg, Yg, Zg = np.meshgrid(x, y, z, indexing='ij')

# Bx, By, Bz, Bmag = Magnetic_Field(
#     Xg, Yg, Zg,
#     X_inner, Y_inner, Z_inner,
#     chunk_size=200  # por ejemplo, trozos de 200
# )
# print("Campo 3D calculado:", Bx.shape)  # (30, 30, 15)
# Uso:

# #wire_plot()
# #Solenoid_points_plot()
# #color_map_B()
Field_view()
# color_map_B()
# Solenoid_points_plot()
# wire_plot()