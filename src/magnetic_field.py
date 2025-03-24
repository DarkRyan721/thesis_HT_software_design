from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
import cupy as cp
from scipy.interpolate import griddata

class B_Field():
    def __init__(self, nSteps=5000, L=0.02, Rin=0.023, Rext=0.05, N=200, I=4.5):
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
        self.Rout = 0.8*self.Rin #[m] -> Radio de los solenoides externos

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

        X1 = xParametrique_outer(phi) + self.Rext + self.Rout/2
        Y1 = yParametrique_outer(phi) + self.Rext + self.Rout/2
        Z1 = zParametrique_outer(phi)
        self.S1 = np.column_stack((X1,Y1,Z1))

        X2 = xParametrique_outer(phi) - self.Rext - self.Rout/2
        Y2 = yParametrique_outer(phi) + self.Rext + self.Rout/2
        Z2 = zParametrique_outer(phi)
        self.S2 = np.column_stack((X2,Y2,Z2))

        X3 = xParametrique_outer(phi) - self.Rext - self.Rout/2
        Y3 = yParametrique_outer(phi) - self.Rext - self.Rout/2
        Z3 = zParametrique_outer(phi)
        self.S3 = np.column_stack((X3,Y3,Z3))

        X4 = xParametrique_outer(phi) + self.Rext + self.Rout/2
        Y4 = yParametrique_outer(phi) - self.Rext - self.Rout/2
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

    def color_map_B(self, S, XY=False, ZY=False, ZX=False, Solenoid_Center = False, All_Solenoids = True, Plane_Value=0, resolution = 100, num_contorn = 30):
        #___________________________________________________________________________________________
        # Grafica del mapa de color segun el plano y la cantidad de solenoides

        if XY == True:
            X = S[:, 0]
            Y = S[:, 1]

            xi = np.linspace(np.min(X), np.max(X), resolution)
            yi = np.linspace(np.min(Y), np.max(Y), resolution)
            eje_x, eje_y = np.meshgrid(xi, yi)

            S_cmp = np.column_stack((eje_x.flatten(), eje_y.flatten(), self.L * np.ones_like(eje_y.flatten())))

        elif ZY == True:
            X = S[:, 2]
            Y = S[:, 1]

            xi = np.linspace(np.min(X), np.max(X), resolution)
            yi = np.linspace(np.min(Y), np.max(Y), resolution)
            eje_x, eje_y = np.meshgrid(xi, yi)

            S_cmp = np.column_stack((Plane_Value * np.ones_like(eje_y.flatten()), eje_y.flatten(), eje_x.flatten()))
        elif ZX==True:
            X = S[:, 2]
            Y = S[:, 0]

            xi = np.linspace(np.min(X), np.max(X), resolution)
            yi = np.linspace(np.min(Y), np.max(Y), resolution)
            eje_x, eje_y = np.meshgrid(xi, yi)

            S_cmp = np.column_stack((eje_y.flatten(), Plane_Value * np.ones_like(eje_y.flatten()), eje_x.flatten()))
        else:
            print("No hay plano seleccionado")
            return
        
        if Solenoid_Center == True:
            B_cmp = self.Magnetic_Field(S=S_cmp, S_solenoid=self.S_Inner)
        elif All_Solenoids == True:
            B_cmp = self.Total_Magnetic_Field(S=S_cmp) 
        else:
            return

        Bx = B_cmp[:, 0].reshape(eje_x.shape)
        By = B_cmp[:, 1].reshape(eje_x.shape)  
        Bz = B_cmp[:, 2].reshape(eje_x.shape) 

        B_values = np.sqrt(Bx**2 + By**2 + Bz**2)
        B_values = np.nan_to_num(B_values, nan=1e-9)
        
        num_levels = num_contorn  
        contour_levels = np.linspace(np.min(B_values[B_values > 0]), np.max(B_values), num_levels)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#131313')

        im = ax.imshow(B_values, cmap='plasma', norm=colors.LogNorm(vmin=np.max([B_values.min(), 1e-6]), vmax=B_values.max()),
                       extent=[eje_x.min(), eje_x.max(), eje_y.min(), eje_y.max()], origin='lower')

        cset = ax.contour(eje_x, eje_y, B_values, levels=contour_levels, linewidths=1, colors='white', norm=colors.LogNorm())
        ax.clabel(cset, inline=True, fmt='%1.2f', fontsize=10)

        # Agregar barra de color con escala logarítmica
        cbar = plt.colorbar(im, ax=ax, label="|B| (Escala Log)")
        cbar.ax.set_facecolor('#283747') 
        cbar.ax.yaxis.set_tick_params(color='white')  # Color de las marcas de la barra
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('|B| (Escala Log)', color='white') 

        # Etiquetas y título
        ax.set_xlabel("Y-axis")
        ax.set_ylabel("Z-axis")
        ax.set_title("Magnitud del Campo Magnético (Escala Log)")
        ax.title.set_color('white')  # Color del título
        ax.xaxis.label.set_color('white')  # Color de la etiqueta del eje X
        ax.yaxis.label.set_color('white')  # Color de la etiqueta del eje Y
        ax.tick_params(axis='x', colors='white')  # Color de las marcas del eje X
        ax.tick_params(axis='y', colors='white')  # Color de las marcas del eje Y

        ax.set_aspect('auto')

        # Mostrar la figura
        plt.show()

    def B_Field_Lines(self, B, S, XY=False, ZY=False, ZX=False, Plane_Value=0, resolution = 100):
        tolerance = 0.001

        if XY == True:
            X = S[:, 0]  # Coordenadas X originales
            Y = S[:, 1]  # Coordenadas Y originales

            Bx = B[:, 0]  # Componente X del campo magnético
            By = B[:, 1]  # Componente Y del campo magnético
        elif ZY == True:
            X = S[:, 2]  # Coordenadas X originales
            Y = S[:, 1]  # Coordenadas Y originales
            Bx = B[:, 2]  # Componente X del campo magnético
            By = B[:, 1]  # Componente Y del campo magnético
        elif ZX==True:
            X = S[:, 2]  # Coordenadas X originales
            Y = S[:, 0]  # Coordenadas Y originales
            Bx = B[:, 2]  # Componente X del campo magnético
            By = B[:, 0]  # Componente Y del campo magnético
        else:
            print("No hay plano seleccionado")
            return
        
        B_mag = np.sqrt(Bx**2 + By**2)

        max_arrow_length = 0.001
        Bx_limited = np.clip(Bx, -max_arrow_length, max_arrow_length)
        By_limited = np.clip(By, -max_arrow_length, max_arrow_length)

        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(8, 8))

        # Graficar las flechas con un mapa de color
        quiver = ax.quiver(X, Y, Bx_limited, By_limited, B_mag, angles='xy', scale_units='xy', scale=0.5, cmap='plasma')

        # Agregar una barra de color para la magnitud
        cbar = plt.colorbar(quiver, ax=ax, label="Magnitud del Campo Magnético")

        # Etiquetas y título
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(np.min(X), np.max(X))
        ax.set_ylim(np.min(Y), np.max(Y))
        ax.set_title('Líneas del Campo Magnético en el Plano XY')

        # Agregar los dos círculos (diámetros 0.056 y 0.1)
        circulo1 = plt.Circle((0, 0), 0.056/2, fill=False, linewidth=2)
        ax.add_patch(circulo1)

        # Mostrar la figura
        plt.show()

    def Save_B_Field(self, B, S):
        MagField_array = np.column_stack((S,B))
        
        np.save("data_files/Magnetic_Field_np.npy", MagField_array)
        print("Archivo guardado")


E_File = np.load("data_files/Electric_Field_np.npy")

spatial_coords = E_File[:, :3]
X = spatial_coords[:, 0]
Y = spatial_coords[:, 1]
Z = spatial_coords[:, 2]

B = B_Field()

B_value = B.Total_Magnetic_Field(S=spatial_coords)

B.Save_B_Field(B=B_value, S=spatial_coords)