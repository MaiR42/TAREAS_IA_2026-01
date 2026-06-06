GUARDAR_IMG = False # Para no demorarme tanton,,, DEBUG

# Inicio tarea 2
import cv2

img1 = cv2.imread("./data/img1.jpg")
img2 = cv2.imread("./data/img2.jpg")
img3 = cv2.imread("./data/img3.jpg")

def resultado_imagen(p_img ,p_sp, p_sr, p_maxLevel):
    resultado_img = cv2.pyrMeanShiftFiltering(
        p_img,
        p_sp,
        p_sr,
        p_maxLevel
    )
    return resultado_img

# Pruebas

imagenes = [img1, img2, img3]
valores_sp = [5, 15, 30] # Elegir 3
valores_sr = [10, 30, 60] # Elegir 3
valores_maxLevel = [0,1]

# Mostrar imagenes originales
def mostrar_imagenes(imagenes):
    i = 0
    for img in imagenes:
        i+=1
        cv2.imshow(f"Original {i}", img)
        input()

if GUARDAR_IMG:
    mostrar_imagenes(imagenes)

# Guardar resultados
import os

os.makedirs("./resultados/img1", exist_ok=True)
os.makedirs("./resultados/img2", exist_ok=True)
os.makedirs("./resultados/img3", exist_ok=True)

if GUARDAR_IMG:
    for num_img, img in enumerate(imagenes, start=1):
        for sp in valores_sp:
            for sr in valores_sr:
                for maxLevel in valores_maxLevel:
                    resultado = resultado_imagen(
                        img,
                        sp,
                        sr,
                        maxLevel
                    )

                    nombre_archivo = (
                        f"./resultados/img{num_img}/"
                        f"sp{sp}_sr{sr}_max{maxLevel}.jpg"
                    )

                    cv2.imwrite(nombre_archivo, resultado)

                    print("Guardado:", nombre_archivo)


# Plot de imagenes para analisis
import matplotlib.pyplot as plt

def comparar_resultados(img, nombre_img, maxLevel):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, sp in enumerate(valores_sp):
        for j, sr in enumerate(valores_sr):
            resultado = resultado_imagen(
                img,
                sp,
                sr,
                maxLevel
            )

            resultado_rgb = cv2.cvtColor(
                resultado,
                cv2.COLOR_BGR2RGB
            )

            axes[i, j].imshow(resultado_rgb)

            axes[i, j].set_title(
                f"sp={sp}, sr={sr}"
            )

            axes[i, j].axis("off")
    plt.suptitle(
        f"{nombre_img} con maxLevel={maxLevel}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


i = 0
for img in imagenes:
    i+=1
    for v_maxLevel in valores_maxLevel:
        texto_plot = f"Imagen {i}"
        comparar_resultados(img, texto_plot, v_maxLevel)



# Nose q hace pero va al final
cv2.waitKey(0)
cv2.destroyAllWindows()