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

cv2.imshow("Original", img1)
cv2.imshow("Mean Shift", resultado_imagen(img1, 15, 30, 0))

cv2.waitKey(0)
cv2.destroyAllWindows()