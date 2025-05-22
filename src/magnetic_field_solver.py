from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
import cupy as cp
from scipy.interpolate import griddata

class B_Field():
    def __init__(self, nSteps=5000, L=0.02, Rin=0.028, Rext=0.05, N=200, I=4.5):
        #___________________________________________________________________________________________
        #   Parametros de los solenoides

        self.nSteps = nSteps #[1] -> Resolucion de un solenoide
        self.L = L #[m] -> Longitud del propulsor/solenoide
        self.Rin = Rin #[m] -> Radio interno
        self.Rext = Rext #[m] -> Radio externo
        self.N = N  #[1] -> Numero de vueltas del solenoide
        self.I = I #[A] -> Corriente en los solenoides

        self.muo = (4e-7)*np.pi #[(T*m)/A] -> Constante de permiabilidad del vacio


        #___________________________________________________________________________________________

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

    def Magnetic_Field(self, S, S_solenoid, chunk_size=100):
        #___________________________________________________________________________________________
        #   Calculo del campo magnetico de un solenoide para los puntos en [S]

        muo = (4e-7)*np.pi #[(T*m)/A] -> Constante de permiabilidad del vacio

        xg = cp.array(S[:,0], dtype=cp.float64)
        yg = cp.array(S[:,1], dtype=cp.float64)
        zg = cp.array(S[:,2], dtype=cp.float64)

        Xs = cp.array(S_solenoid[:,0], dtype=cp.float64)
        Ys = cp.array(S_solenoid[:,1], dtype=cp.float64)
        Zs = cp.array(S_solenoid[:,2], dtype=cp.float64)

        # Inicializar acumuladores en la GPU
        Bx = cp.zeros_like(xg, dtype=cp.float64)
        By = cp.zeros_like(yg, dtype=cp.float64)
        Bz = cp.zeros_like(zg, dtype=cp.float64)

        # Segmentos diferenciales del solenoide
        dlx = Xs[1:] - Xs[:-1]
        dly = Ys[1:] - Ys[:-1]
        dlz = Zs[1:] - Zs[:-1]

        for start in tqdm(range(0, self.nSteps - 1, chunk_size), desc="Calculando campo magnético", unit="chunk"):
            end = min(start + chunk_size, self.nSteps - 1)

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

        factor = (muo * self.I) / (4.0 * cp.pi)
        Bx *= factor
        By *= factor
        Bz *= factor

        B_coord = np.column_stack((Bx.get(), By.get(), Bz.get()))
        return B_coord #(T)
    
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

    B_field.B_Field_Lines(ZX=True, Plane_Value=0.0, All_Solenoids=True, Solenoid_Center=False)
    #B_field.B_Field_Heatmap(XY=False, ZX=True, Plane_Value=0.0, Solenoid_Center=False, All_Solenoids=True)

    # 4. Guardar en el archivo el campo magnetico encontrado

    #B_field.Save_B_Field(B=B_value, S=spatial_coords)

    # 5.Diferentes opciones de plot para el usuario(Opcional)

    #B_field.color_map_B(S=spatial_coords, XY=True, Plane_Value=0.01, num_contorn=10, resolution=400, Solenoid_Center=True)

    #B_field.Solenoid_points_plot(Solenoid_1=True, Solenoid_2=True, Solenoid_3=True, Solenoid_4=True)