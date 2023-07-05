import cv2 as cv 
import numpy as np
import random as rd

part = "./22962.jpg"

img = cv.imread(part,cv.IMREAD_GRAYSCALE)
img = np.array(img,dtype="uint8")

density_salt = 0.2
density_pepper = 0.2

number_of_white_pexel = int(density_salt * img.shape[0] * img.shape[1])
number_of_black_pexel = int(density_pepper * img.shape[0] * img.shape[1])
img_noise=np.array(img,dtype="uint8")
#เติม nois ขาว
for i in range(number_of_white_pexel):
    y=rd.randint(0,img.shape[0]-1)
    x=rd.randint(0,img.shape[1]-1)
    img_noise[y][x]=255
    
#เติม noise ดำ
for i in range(number_of_white_pexel):
    y=rd.randint(0,img.shape[0]-1)
    x=rd.randint(0,img.shape[1]-1)
    img_noise[y][x]=0

# ขั้น denoise
denoised_img = cv.medianBlur(img_noise, 7)

cv.imwrite("./denoised.png",denoised_img)

#noise ที่ไม่ถูกลบหาได้จาก out-in เเบบ อ.เเจ็ค สอน
out_vs_in = denoised_img-img
cv.imwrite("./noised.png",img_noise)
cv.imwrite("./original.png",img)
cv.imwrite("./out_vs_in.png",out_vs_in)