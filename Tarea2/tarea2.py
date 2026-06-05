# Inicio tarea 2
import cv2

img1 = cv2.imread("./data/img1.jpg")
img2 = cv2.imread("./data/img2.jpg")
img3 = cv2.imread("./data/img3.jpg")

def resultado_imagen(p_img ,p_sp, p_sr, p_maxLevel):
    resultado_img = cv2.pyrMeanShiftFiltering(
        img=p_img,
        sp=p_sp,
        sr=p_sr,
        p_maxLevel=0
    )
    return resultado_img

cv2.waitKey(0)
cv2.destroyAllWindows()