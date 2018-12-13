import cv2
import numpy as np



# filter = True


def erode(img: np.ndarray, kernelShape: tuple) -> np.ndarray:
    kernel = np.ones(kernelShape,np.uint8)
    return cv2.erode(img, kernel, iterations = 1)

def dilate(img: np.ndarray, kernelShape: tuple) -> np.ndarray:
    kernel = np.ones(kernelShape,np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

def gaussianblur(img: np.ndarray, ksize: tuple) -> np.ndarray:
    return cv2.GaussianBlur(img, ksize, 0)

def opening(img: np.ndarray, kernelShape: tuple) -> np.ndarray:
    kernel = np.ones(kernelShape,np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

file_path = 'a.jpg'
img_org = cv2.imread(file_path)

img = cv2.imread(file_path, 0)

gray_org_img = img

height, width = img.shape[:2]

if height > 10000 or width > 10000:
    exit(1)

i = 5

while i > 0:
    img = gray_org_img
    img = dilate(img, (i, i))

    cv2.imwrite('b.png',img)

    blur = cv2.GaussianBlur(img,(5,5),0)

    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    # ret3,th3 = cv2.threshold(th3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # th3 = opening(th3, (5, 5))



    imgEdge,contours,hierarchy = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours.sort(key=cv2.contourArea, reverse=True)

    contours = contours[:5]

    maxArea = height * width
    
    for cnt in contours:
        sqeeze = np.squeeze(cnt)

        xMin = min(sqeeze[:, 0])
        xMax = max(sqeeze[:, 0])
        yMin = min(sqeeze[:, 1])
        yMax = max(sqeeze[:, 1])

        if xMin == 0 or xMax == width - 1 or yMin == 0 or yMax == height -1:
            continue

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        k = cv2.isContourConvex(cnt)
        # print(area, perimeter, k)
        # print(i)
        # if (area > maxArea * 0.3) and (maxArea * 0.95 > area):
        if area >= maxArea/2 and area <= maxArea:
            # epsilon = 0.01 * perimeter
            # approx = cv2.approxPolyDP(cnt,epsilon,True)
            # cv2.drawContours(img_org,[approx],-1,[0,255,0],2)
            i = 0
            print(xMin, xMax, yMin, yMax)
            p1=[-1, -1]
            p2=[-1, -1]
            p3=[-1, -1]
            p4=[-1, -1]
            firstIn = -1
            for v in np.squeeze(cnt):
                # print(v)
                if v[0] == xMin:
                    p1[0], p1[1] = v
                    firstIn = 1
                    print(1)
                elif v[1] == yMin:
                    p2[0], p2[1] = v
                    print(2)
                elif v[1] == yMax:
                    p3[0], p3[1] = v
                    print(3)
                elif v[0] == xMax:
                    p4[0], p4[1] = v
                    firstIn = 4
                    print(4)
            print(p1, p2, p3, p4)
            pts2 = np.float32([[0,0],[height,0],[0,width],[height,width]])

            if p1[1] < p4[1]:
                pts1 = np.float32([p1,p2,p3,p4])
            else:
                pts1 = np.float32([p2,p4,p1,p3])

            M = cv2.getPerspectiveTransform(pts1,pts2)
            rst = cv2.warpPerspective(img_org,M,(height,width))
            break
    i -= 1

cv2.imwrite('original.jpg',img_org)
cv2.imwrite('o.jpg',rst)
