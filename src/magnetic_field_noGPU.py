from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
<<<<<<< HEAD
from numpy import sin, cos, pi

=======
from scipy.interpolate import griddata
>>>>>>> f0c1ecc311a500d8381ea1c7e228eeefcc7346ba

class B_Field():
    def __init__(self, nSteps=5000, L=0.02, Rin=0.028, Rext=0.05, N=200, I=4.5):
        #___________________________________________________________________________________________
        #   Parametros de los solenoides

<<<<<<< HEAD
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
=======
        self.nSteps = nSteps #[1] -> Resolucion de un solenoide
        self.L = L #[m] -> Longitud del propulsor/solenoide
        self.Rin = Rin #[m] -> Radio interno
        self.Rext = Rext #[m] -> Radio externo
        self.N = N  #[1] -> Numero de vueltas del solenoide
        self.I = I #[A] -> Corriente en los solenoides

        self.muo = (4e-7)*np.pi #[(T*m)/A] -> Constante de permiabilidad del vacio

        #___________________________________________________________________________________________
>>>>>>> f0c1ecc311a500d8381ea1c7e228eeefcc7346ba

        self.Solenoid_Create()

    def Solenoid_Create(self):
        #___________________________________________________________________________________________
        #   Creacion de los solenoides

        phi = np.linspace(0, 2*np.pi, self.nSteps)
        self.Rout = 0.5*self.Rin #[m] -> Radio de los solenoides externos
        Rout_center = self.Rext

        def xParametrique_inner( phi ) : return self.Rin*np.cos(self.N*phi)
        def yParametrique_inner( phi ) : return self.Rin*np.sin(self.N*phi)
        def zParametrique_inner( phi ) : return self.L*phi/(2*np.pi)

        def xParametrique_outer( phi ) : return self.Rout*np.cos(self.N*phi)
        def yParametrique_outer( phi ) : return self.Rout*np.sin(self.N*phi)
        def zParametrique_outer( phi ) : return self.L*phi/(2*np.pi)

        X_inner = xParametrique_inner(phi)
        Y_inner = yParametrique_inner(phi)
        Z_inner = zParametrique_inner(phi)
        self.S_Inner = np.column_stack((X_inner,Y_inner,Z_inner))

        X1 = xParametrique_outer(phi) + Rout_center
        Y1 = yParametrique_outer(phi) + Rout_center
        Z1 = zParametrique_outer(phi)
        self.S1 = np.column_stack((X1,Y1,Z1))

        X2 = xParametrique_outer(phi) - Rout_center
        Y2 = yParametrique_outer(phi) + Rout_center
        Z2 = zParametrique_outer(phi)
        self.S2 = np.column_stack((X2,Y2,Z2))

        X3 = xParametrique_outer(phi) - Rout_center
        Y3 = yParametrique_outer(phi) - Rout_center
        Z3 = zParametrique_outer(phi)
        self.S3 = np.column_stack((X3,Y3,Z3))

        X4 = xParametrique_outer(phi) + Rout_center
        Y4 = yParametrique_outer(phi) - Rout_center
        Z4 = zParametrique_outer(phi)
        self.S4 = np.column_stack((X4,Y4,Z4))

        #___________________________________________________________________________________________

    def Magnetic_Field(S, S_solenoid, nSteps, current=4.5, chunk_size=100):
        muo = 4e-7 * np.pi  # permeabilidad del vacío

        xg = S[:, 0]
        yg = S[:, 1]
        zg = S[:, 2]

        Xs = S_solenoid[:, 0]
        Ys = S_solenoid[:, 1]
        Zs = S_solenoid[:, 2]

<<<<<<< HEAD
    # 4) Devolver a CPU
    # return Bx.get(), By.get(), Bz.get(), B_mag.get()
    return Bx, By, Bz, B_mag
=======
        Bx = np.zeros_like(xg, dtype=np.float64)
        By = np.zeros_like(yg, dtype=np.float64)
        Bz = np.zeros_like(zg, dtype=np.float64)
>>>>>>> f0c1ecc311a500d8381ea1c7e228eeefcc7346ba

        dlx = Xs[1:] - Xs[:-1]
        dly = Ys[1:] - Ys[:-1]
        dlz = Zs[1:] - Zs[:-1]

        for start in tqdm(range(0, nSteps - 1, chunk_size), desc="Campo magnético (NumPy)", unit="chunk"):
            end = min(start + chunk_size, nSteps - 1)

            dlx_chunk = dlx[start:end]
            dly_chunk = dly[start:end]
            dlz_chunk = dlz[start:end]

            X_sol_chunk = Xs[start:end]
            Y_sol_chunk = Ys[start:end]
            Z_sol_chunk = Zs[start:end]

            for k in range(len(dlx_chunk)):
                rx = xg - X_sol_chunk[k]
                ry = yg - Y_sol_chunk[k]
                rz = zg - Z_sol_chunk[k]

                rnorm = np.sqrt(rx**2 + ry**2 + rz**2)
                rnorm = np.maximum(rnorm, 1e-14)

                dBx = (dly_chunk[k] * rz - dlz_chunk[k] * ry) / (rnorm**3)
                dBy = (dlz_chunk[k] * rx - dlx_chunk[k] * rz) / (rnorm**3)
                dBz = (dlx_chunk[k] * ry - dly_chunk[k] * rx) / (rnorm**3)

                Bx += dBx
                By += dBy
                Bz += dBz

        factor = (muo * current) / (4.0 * np.pi)
        Bx *= factor
        By *= factor
        Bz *= factor

        return np.column_stack((Bx, By, Bz))
    
        #___________________________________________________________________________________________
    
    def Total_Magnetic_Field(self, S, chunk_size=100):
        #___________________________________________________________________________________________
        #   Calculo total del campo magnetico para los puntos [S]
        
        B_inner = self.Magnetic_Field(S=S, S_solenoid=self.S_Inner, chunk_size=chunk_size)
        B1 = self.Magnetic_Field(S=S, S_solenoid=self.S1, chunk_size=chunk_size)
        B2 = self.Magnetic_Field(S=S, S_solenoid=self.S2, chunk_size=chunk_size)
        B3 = self.Magnetic_Field(S=S, S_solenoid=self.S3, chunk_size=chunk_size)
        B4 = self.Magnetic_Field(S=S, S_solenoid=self.S4, chunk_size=chunk_size)

        B_total = B_inner - (B1+B2+B3+B4)

        return B_total #(Bx,By,Bz)

        #___________________________________________________________________________________________

    def Solenoid_points_plot(self, Solenoid_inner=True, Solenoid_1=True, Solenoid_2=True, Solenoid_3=True, Solenoid_4=True, Cylinder_ext=True):

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        if Solenoid_inner == True:
            ax.scatter(self.S_Inner[:,0], self.S_Inner[:,1], self.S_Inner[:,2], "black")
        if Solenoid_1 == True:
            ax.scatter(self.S1[:,0], self.S1[:,1], self.S1[:,2], "black")
        if Solenoid_2 == True:
            ax.scatter(self.S2[:,0], self.S2[:,1], self.S2[:,2], "black")
        if Solenoid_3 == True:
            ax.scatter(self.S3[:,0], self.S3[:,1], self.S3[:,2], "black")
        if Solenoid_4 == True:
            ax.scatter(self.S4[:,0], self.S4[:,1], self.S4[:,2], "black")
        if Cylinder_ext == True:
            # Crear la malla de puntos
            theta = np.linspace(0, 2 * np.pi, 100)  # Ángulo para la circunferencia
            z = np.linspace(0, self.L, 50)               # Longitud del cilindro
            theta, z = np.meshgrid(theta, z)         # Crear una malla de ángulos y alturas

            # Convertir coordenadas cilíndricas a cartesianas
            x = self.Rext * np.cos(theta)
            y = self.Rext * np.sin(theta)

            ax.plot_surface(x, y, z, color='black', alpha=0.7)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect('equal')

        plt.show()

    def B_Field_Heatmap(self, Solenoid_Center=False, All_Solenoids=False, XY=False, ZX=False, Plane_Value=0.0, resolution=100):
        if XY:
            # 1. Crear malla estructurada en XY
            xi = np.linspace(-1.5 * self.Rext, 1.5 * self.Rext, resolution)
            yi = np.linspace(-1.5 * self.Rext, 1.5 * self.Rext, resolution)
            Xi, Yi = np.meshgrid(xi, yi)

            # 2. Expandir a coordenadas 3D con Z constante = Plane_Value
            S_eval = np.column_stack((Xi.ravel(), Yi.ravel(), np.full(Xi.size, Plane_Value)))
        elif ZX:
            # 1. Crear malla estructurada en ZX
            xi = np.linspace(0.0, 1.5 * self.L, resolution)  # eje Z
            yi = np.linspace(-1.5 * self.Rext, 1.5 * self.Rext, resolution)  # eje X
            Xi, Yi = np.meshgrid(xi, yi)

            # 2. Expandir a coordenadas 3D con Y constante = Plane_Value
            S_eval = np.column_stack((Yi.ravel(), np.full(Xi.size, Plane_Value), Xi.ravel()))
        else:
            print("❌ Debes seleccionar un plano de plot.")
            return

        # 3. Evaluar campo magnético en esa malla
        if Solenoid_Center:
            B_eval = self.Magnetic_Field(S=S_eval, S_solenoid=self.S_Inner)
        elif All_Solenoids:
            B_eval = self.Total_Magnetic_Field(S=S_eval)
        else:
            print("❌ Debes escoger el o los solenoides presentes en el gráfico.")
            return

        # 4. Separar componentes del campo
        if XY:
            Bx = B_eval[:, 0].reshape(Xi.shape)
            By = B_eval[:, 1].reshape(Yi.shape)

            Bmag = np.sqrt(Bx**2 + By**2)
            R = np.sqrt(Xi**2 + Yi**2)
            Bmag[R > self.Rext] = Bmag[Bmag > 1e-8].min()
        elif ZX:
            Bx = B_eval[:, 2].reshape(Xi.shape)  # Bz (horizontal)
            By = B_eval[:, 0].reshape(Yi.shape)  # Bx (vertical)

            Bmag = np.sqrt(Bx**2 + By**2)
        
        vmin = Bmag[Bmag > 1e-8].min()
        vmax = Bmag.max()
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 300)

        cmap = plt.get_cmap('inferno').copy()
        cmap.set_under('black')

        # 5. Graficar solo mapa de calor
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.contourf(
            Xi, Yi, Bmag,
            levels=levels,
            cmap=cmap,
            norm=colors.LogNorm(vmin=vmin, vmax=vmax)
        )

        cbar = plt.colorbar(heatmap, ax=ax, ticks=np.linspace(vmin, vmax, 5))
        cbar.ax.yaxis.set_major_formatter('{:.2e}'.format)  # o usa '%.2e'
        cbar.set_label("Magnitud del Campo Magnético")

        # 6. Etiquetas
        if XY:
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Mapa de Calor del Campo Magnético - Plano XY")
            ax.set_xlim(-1.5 * self.Rext, 1.5 * self.Rext)
            ax.set_ylim(-1.5 * self.Rext, 1.5 * self.Rext)

            # 6. Dibujar geometría del canal
            inner_circle = plt.Circle((0, 0), self.Rin, fill=True, linewidth=2, facecolor='gray')
            outer_circle = plt.Circle((0, 0), self.Rext, fill=False, linewidth=4, edgecolor='gray')
            ax.add_patch(inner_circle)
            ax.add_patch(outer_circle)
        elif ZX:
            ax.set_xlabel("Z [m]")
            ax.set_ylabel("X [m]")
            ax.set_title("Mapa de Calor del Campo Magnético - Plano ZX")
            ax.set_xlim(0.0, 1.5 * self.L)
            ax.set_ylim(-1.5 * self.Rext, 1.5 * self.Rext)

            # 7. Estética
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Líneas del Campo Magnético - Plano ZX")
            ax.set_xlim(0.0, 1.5 * self.L)
            ax.set_ylim(-1.5 * self.Rext, 1.5 * self.Rext)

            rect_sup = plt.Rectangle(
                (0.0, self.Rext),
                self.L,
                0.5*self.Rext,
                linewidth=2,
                edgecolor='none',
                facecolor='gray'
            )
            rect_inf = plt.Rectangle(
                (0.0, -1.5*self.Rext),
                self.L,
                0.5*self.Rext,
                linewidth=2,
                edgecolor='none',
                facecolor='gray'
            )
            rect_inner = plt.Rectangle(
                (0.0, -self.Rin),
                self.L,
                2*self.Rin,
                linewidth=2,
                edgecolor='none',
                facecolor='gray'
            )

            ax.add_patch(rect_sup)
            ax.add_patch(rect_inf)
            ax.add_patch(rect_inner)

        plt.tight_layout()
        plt.show()

    def B_Field_Lines(self, Solenoid_Center=False, All_Solenoids=False, XY=False, ZX=False, Plane_Value=0.0, resolution=100):
        if XY:
            # 1. Crear malla estructurada en XY
            xi = np.linspace(-1.5 * self.Rext, 1.5 * self.Rext, resolution)
            yi = np.linspace(-1.5 * self.Rext, 1.5 * self.Rext, resolution)
            Xi, Yi = np.meshgrid(xi, yi)

            # 2. Expandir a coordenadas 3D con Z constante = Plane_Value
            S_eval = np.column_stack((Xi.ravel(), Yi.ravel(), np.full(Xi.size, Plane_Value)))
        elif ZX:
            # 1. Crear malla estructurada en ZX
            xi = np.linspace(0.0, 1.5 * self.L, resolution) # eje Z
            yi = np.linspace(-1.5 * self.Rext, 1.5 * self.Rext, resolution) # eje X
            Xi, Yi = np.meshgrid(xi, yi)

            # 2. Expandir a coordenadas 3D con Z constante = Plane_Value
            S_eval = np.column_stack((Yi.ravel(), np.full(Xi.size, Plane_Value) ,Xi.ravel()))
        else:
            print("❌ Debes seleccionar un plano de plot.")
            return

        # 3. Evaluar campo magnético en esa malla
        if Solenoid_Center:
            B_eval = self.Magnetic_Field(S=S_eval, S_solenoid=self.S_Inner)
        elif All_Solenoids:
            B_eval = self.Total_Magnetic_Field(S=S_eval)
        else:
            print("❌ Debes escoger el o los solenoides presentes en el grafico.")
            return
        
        if XY:
            # 4. Separar componentes del campo y reordenar a la forma de malla
            Bx = B_eval[:, 0].reshape(Xi.shape)
            By = B_eval[:, 1].reshape(Yi.shape)
        elif ZX:
            Bx = B_eval[:, 2].reshape(Xi.shape)  # Bz (horizontal)
            By = B_eval[:, 0].reshape(Yi.shape)  # Bx (vertical)

        Bmag = np.sqrt(Bx**2 + By**2)

        # 5. Graficar con fondo de magnitud + líneas de flujo
        fig, ax = plt.subplots(figsize=(10, 8))
        color_plot = ax.contourf(Xi, Yi, Bmag, levels=100, cmap='viridis')

        ax.streamplot(Xi, Yi, Bx, By, color='white', linewidth=0.7, density=1, arrowsize=0.7)

        cbar = plt.colorbar(color_plot, ax=ax)
        cbar.set_label("Magnitud del Campo Magnético")

        if XY:
            # 7. Estética
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Líneas del Campo Magnético - Plano XY")
            ax.set_xlim(-1.5 * self.Rext, 1.5 * self.Rext)
            ax.set_ylim(-1.5 * self.Rext, 1.5 * self.Rext)


            # 6. Dibujar geometría del canal
            inner_circle = plt.Circle((0, 0), self.Rin, fill=True, linewidth=2, facecolor='black')
            outer_circle = plt.Circle((0, 0), self.Rext, fill=False, linewidth=4, edgecolor='black')
            ax.add_patch(inner_circle)
            ax.add_patch(outer_circle)
        elif ZX:
            # 7. Estética
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Líneas del Campo Magnético - Plano ZX")
            ax.set_xlim(0.0, 1.5 * self.L)
            ax.set_ylim(-1.5 * self.Rext, 1.5 * self.Rext)

            rect_sup = plt.Rectangle(
                (0.0, self.Rext),
                self.L,
                0.5*self.Rext,
                linewidth=2,
                edgecolor='none',
                facecolor='gray'
            )
            rect_inf = plt.Rectangle(
                (0.0, -1.5*self.Rext),
                self.L,
                0.5*self.Rext,
                linewidth=2,
                edgecolor='none',
                facecolor='gray'
            )
            rect_inner = plt.Rectangle(
                (0.0, -self.Rin),
                self.L,
                2*self.Rin,
                linewidth=2,
                edgecolor='none',
                facecolor='gray'
            )

            ax.add_patch(rect_sup)
            ax.add_patch(rect_inf)
            ax.add_patch(rect_inner)

        plt.tight_layout()
        plt.show()

    def Save_B_Field(self, B, S):
        MagField_array = np.column_stack((S,B))
        
        np.save("data_files/Magnetic_Field_np.npy", MagField_array)
        print("Archivo guardado")

if __name__ == "__main__":

    # EXAMPLE FOR GUI:

    # 1. Cargar y guardar los nodos de la malla

    E_File = np.load("data_files/Electric_Field_np.npy")
    spatial_coords = E_File[:, :3]

    # 2. Crear objeto del campo magnetico

    """
        Entradas que brinda el usuario para el campo magentico:

        nSteps -> Resolucion de un solenoide (afecta el rendimiento)
        L -> Longitud o profundidad del propulsor/solenoide
        Rin -> Radio interno
        Rext -> Radio externo
        N -> Numero de vueltas del solenoide (opcional)
        I -> Corriente en los solenoides (opcional)
    """

    B_field = B_Field(nSteps=20000)

    # 3. Calcular el campo magnetico total producido por los 5 solenoides

    #B_value = B_field.Magnetic_Field(S=spatial_coords, S_solenoid=B_field.S_Inner)
    #B_value = B_field.Total_Magnetic_Field(S=spatial_coords)

    #B_field.B_Field_Lines(ZX=True, Plane_Value=0.0, All_Solenoids=True, Solenoid_Center=True)
    B_field.B_Field_Heatmap(XY=True, ZX=False, Plane_Value=0.01, Solenoid_Center=True, All_Solenoids=True)

    # 4. Guardar en el archivo el campo magnetico encontrado

    #B_field.Save_B_Field(B=B_value, S=spatial_coords)

    # 5.Diferentes opciones de plot para el usuario(Opcional)

<<<<<<< HEAD
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
=======
    #B_field.color_map_B(S=spatial_coords, XY=True, Plane_Value=0.01, num_contorn=10, resolution=400, Solenoid_Center=True)

    #B_field.Solenoid_points_plot(Solenoid_1=True, Solenoid_2=True, Solenoid_3=True, Solenoid_4=True)
>>>>>>> f0c1ecc311a500d8381ea1c7e228eeefcc7346ba
