from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors

#_____________________________________________________________________________________________________________
#                   1] Parametros iniciales del campo magnetico

# Numero de partes del solenoide
nSteps = 2500 

# Longitud, Radio y # vueltas del solenoide
L = 0.021 #[m]
Rin = 0.027 #[m]
Rout = 0.8*Rin
Rext = 0.05
N = 200  #[1]

# Constante de permiabilidad del vacio y corriente del solenoides
muo = (4e-7)*np.pi #[(T*m)/A]
i = 4.5 #[A]

#_____________________________________________________________________________________________________________
#                   2] Funcion que calcula la magnitud del campo en un punto (X,Y,Z)
import cupy as cp
def Magnetic_Field(x_grid, y_grid, z_grid, X_sol, Y_sol, Z_sol, chunk_size=100):
    # Asegurar que las salidas tengan la misma forma que la malla de entrada
    # 1) Convertir todo a cupy
    xg = cp.array(x_grid, dtype=cp.float64)
    yg = cp.array(y_grid, dtype=cp.float64)
    zg = cp.array(z_grid, dtype=cp.float64)

    Xs = cp.array(X_sol, dtype=cp.float64)
    Ys = cp.array(Y_sol, dtype=cp.float64)
    Zs = cp.array(Z_sol, dtype=cp.float64)

    # Inicializar acumuladores en la GPU
    Bx = cp.zeros_like(xg, dtype=cp.float64)
    By = cp.zeros_like(yg, dtype=cp.float64)
    Bz = cp.zeros_like(zg, dtype=cp.float64)

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

            rnorm = cp.sqrt(rx*rx + ry*ry + rz*rz)
            # Evitar dividir por cero
            rnorm = cp.maximum(rnorm, 1e-14)

            # dl x r
            dBx = (dly_chunk[k] * rz - dlz_chunk[k] * ry) / (rnorm**3)
            dBy = (dlz_chunk[k] * rx - dlx_chunk[k] * rz) / (rnorm**3)
            dBz = (dlx_chunk[k] * ry - dly_chunk[k] * rx) / (rnorm**3)

            # Acumular al campo total
            Bx += dBx
            By += dBy
            Bz += dBz

        # (Opcional) Sincronizar el GPU para que tqdm no avance antes de tiempo
        # cp.cuda.Stream.null.synchronize()

    # 3) Aplicar la constante de Biot-Savart
    factor = (muo * i) / (4.0 * cp.pi)
    Bx *= factor
    By *= factor
    Bz *= factor

    # Magnitud
    B_mag = cp.sqrt(Bx**2 + By**2 + Bz**2)

    # 4) Devolver a CPU
    return Bx.get(), By.get(), Bz.get(), B_mag.get()

#_____________________________________________________________________________________________________________
#                       3] Creacion de puntos del solenoide


phi = np.linspace(0, 2*np.pi, nSteps)

def xParametrique_inner( phi ) : return Rin*np.cos(N*phi)
def yParametrique_inner( phi ) : return Rin*np.sin(N*phi)
def zParametrique_inner( phi ) : return L*phi/(2*np.pi)

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


def Field_view():
    x_grid = np.linspace(-0.08, 0.08, 40)  # Limita el rango para la visualización
    y_grid = np.linspace(-0.08, 0.08, 40)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)

    # Generar las componentes del campo magnético en el plano XY
    # Aquí asumo que tienes la función 'Magnetic_Field' ya definida
    # Calculamos las componentes del campo en el plano XY usando los valores de X y Y
    Bx_inner, By_inner, Bz_inner, B_mag = Magnetic_Field(X, Y, Z, X_inner, Y_inner, Z_inner)
    Bx1, By1, Bz1, _ = Magnetic_Field(X, Y, Z, X1, Y1, Z1)
    Bx2, By2, Bz1, _ = Magnetic_Field(X, Y, Z, X2, Y2, Z2)
    Bx3, By3, Bz1, _ = Magnetic_Field(X, Y, Z, X3, Y3, Z3)
    Bx4, By4, Bz1, _ = Magnetic_Field(X, Y, Z, X4, Y4, Z4)

    Bx = Bx_inner - (Bx1+Bx2+Bx3+Bx4)
    By = By_inner - (By1+By2+By3+By4)

    max_arrow_length = 0.003  # Longitud máxima de las flechas
    Bx_limited = np.clip(Bx, -max_arrow_length, max_arrow_length)
    By_limited = np.clip(By, -max_arrow_length, max_arrow_length)

    # Plotear el campo magnético en el plano XY usando flechas
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(X, Y, Bx_limited, By_limited, angles='xy', scale_units='xy', scale=1, color='b')

    # Etiquetas y título
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Líneas del Campo Magnético en el Plano XY')

    plt.show()

#wire_plot()
#Solenoid_points_plot()
color_map_B()
#Field_view()