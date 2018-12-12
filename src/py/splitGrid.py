import cv2
import numpy as np

# カレンダー
img = cv2.imread("sample.jpg")
img2 = img.copy()
img3 = img.copy()

# グレースケール
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("calendar_mod.jpg", gray)

gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, (2,2))
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, (2, 2))
cv2.imwrite("calendar_mod5.jpg", gray)


## 反転 ネガポジ変換
gray2 = cv2.bitwise_not(gray)
cv2.imwrite("calendar_mod2.jpg", gray2)

edges = cv2.Canny(gray2,50,150,apertureSize = 3)

lines = cv2.HoughLinesP(edges,1,np.pi/180,100)


for line in lines:
    x1, y1, x2, y2 = line[0]

    # 赤線を引く
    red_lines_img = cv2.line(img2, (x1,y1), (x2,y2), (0,0,255), 3)
    cv2.imwrite("calendar_mod3.jpg", red_lines_img)

    # 線を消す(白で線を引く)
    no_lines_img = cv2.line(img3, (x1,y1), (x2,y2), (255,255,255), 3)
    cv2.imwrite("calendar_mod4.jpg", no_lines_img)