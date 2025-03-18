from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#_____________________________________________________________________________________________________________
#                   1] Parametros iniciales del campo magnetico

# Numero de partes del solenoide
nSteps = 300 

# Longitud, Radio y # vueltas del solenoide
L = 60 #[m]
R = 20 #[m]
N = 10  #[1]

# Constante de permiabilidad del vacio y corriente del solenoides
muo = 1 #[(T*m)/A]
i = 1 #[A]

#_____________________________________________________________________________________________________________
#                   2] Funcion que calcula la magnitud del campo en un punto (X,Y,Z)

def Magnetic_Field(x_grid, y_grid, z_grid, X_sol, Y_sol, Z_sol):
    # Asegurar que las salidas tengan la misma forma que la malla de entrada
    Bx = np.zeros_like(x_grid, dtype=np.float64)
    By = np.zeros_like(y_grid, dtype=np.float64)
    Bz = np.zeros_like(z_grid, dtype=np.float64)

    # Diferenciales del solenoide
    dl_x = X_sol[1:] - X_sol[:-1]
    dl_y = Y_sol[1:] - Y_sol[:-1]
    dl_z = Z_sol[1:] - Z_sol[:-1]
    
    for n in tqdm(range(nSteps - 1), desc="Calculando campo magnético", unit="paso"):
        r_x = x_grid - X_sol[n]
        r_y = y_grid - Y_sol[n]
        r_z = z_grid - Z_sol[n]

        r_norm = np.sqrt(r_x**2 + r_y**2 + r_z**2)

        # Evitar divisiones por cero
        r_norm[r_norm == 0] = 1e-9

        # Producto cruzado dl x r (calculado por componentes)
        dB_x = (dl_y[n] * r_z - dl_z[n] * r_y) / r_norm**3
        dB_y = (dl_z[n] * r_x - dl_x[n] * r_z) / r_norm**3
        dB_z = (dl_x[n] * r_y - dl_y[n] * r_x) / r_norm**3

        # Sumar contribución al campo total
        Bx += dB_x
        By += dB_y
        Bz += dB_z

    # Aplicar la constante de Biot-Savart
    Bx *= (muo * i) / (4 * np.pi)
    By *= (muo * i) / (4 * np.pi)
    Bz *= (muo * i) / (4 * np.pi)

    # Retorna la magnitud del campo
    return Bx, By, Bz, np.sqrt(Bx**2 + By**2 + Bz**2)

#_____________________________________________________________________________________________________________
#                       3] Creacion de puntos del solenoide


phi = np.linspace(0, 2*np.pi, nSteps)

def xParametrique( phi ) : return R*np.cos(N*phi)
def yParametrique( phi ) : return R*np.sin(N*phi)
def zParametrique( phi ) : return L*phi/(2*np.pi)

X1 = xParametrique(phi)
Y1 = yParametrique(phi) - 40
Z1 = zParametrique(phi)

X2 = xParametrique(phi)
Y2 = yParametrique(phi) + 40
Z2 = zParametrique(phi)

#_____________________________________________________________________________________________________________
#                       4] Funcion para ploteo de los solenoides

def Solenoid_plot():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # ax.plot3D(X, Y, Z, 'black')
    ax.scatter(X1, Y1, Z1, 'black')
    ax.scatter(X2, Y2, Z2, "black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect('equal')

    plt.show()

#_____________________________________________________________________________________________________________
#                       5] Funcion para vista lateral del campo magnetico

def color_map_B():
    # Generar la malla de evaluación
    yVal_focus = np.linspace(-100, 100, 500)  # Reducir el rango en Y
    zVal_focus = np.linspace(-30, L+30, 500)  # Mantener Z igual

    # Crear la malla de puntos
    YVec_focus, ZVec_focus = np.meshgrid(yVal_focus, zVal_focus)

    # Calcular la magnitud del campo en la malla
    _,_,_,B1 = Magnetic_Field(0.001 * np.ones_like(YVec_focus), YVec_focus, ZVec_focus, X1, Y1, Z1) 
    _,_,_,B2 = Magnetic_Field(0.001 * np.ones_like(YVec_focus), YVec_focus, ZVec_focus, X2, Y2, Z2)

    B_values = np.abs(B1 - B2)

    num_levels = 10  
    contour_levels = np.linspace(np.min(B_values[B_values > 0]), np.max(B_values), num_levels)

    import matplotlib.colors as colors

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
    yVal = np.linspace(-100, 100, 300)
    zVal = np.linspace(-30, L+30, 300)
    YVec, ZVec = np.meshgrid(yVal, zVal)

    # Calcular el campo magnético en la malla
    Bx1, By1, Bz1, Bmag1 = Magnetic_Field(np.ones_like(YVec), YVec, ZVec, X1, Y1, Z1)
    Bx2, By2, Bz2, Bmag2  = Magnetic_Field(np.ones_like(YVec), YVec, ZVec, X2, Y2, Z2)

    By = By1 - By2
    Bz = Bz1 - Bz2

    # Visualizar las líneas de campo magnético usando streamplot y quiver
    fig, ax = plt.subplots(figsize=(10, 10))

    # Graficar las líneas de campo magnético usando streamplot
    ax.streamplot(YVec, ZVec, By, Bz, density=2, color='blue')

    # Graficar las flechas del campo magnético usando quiver
    ax.quiver(YVec, ZVec, By, Bz, color='black')

    # Ajustar la apariencia
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')
    ax.axis([-100, 100, -30, L+30])

    plt.show()

#Solenoid_plot()
Field_view()