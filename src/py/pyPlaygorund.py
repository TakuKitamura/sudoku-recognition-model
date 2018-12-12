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

file_path = '2.jpg'
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
        # print(max(np.squeeze(cnt)[0]))

        xMaxList = []
        xMinList = []
        yMaxList = []
        yMinList = []
        # for v in np.squeeze(cnt):
        #     if (np.array([0, 0]) == v).all():
        #         print("ok1")
        #     elif (np.array([height - 1, width - 1]) == v).all():
        #         print("ok2")

        # print(cnt[:, 0])
        # xMaxList.append(max(cnt[:, 0]))
        # xMinList.append(min(cnt[:, 0]))

        # yMaxList.append(max(cnt[:, 1]))
        # yMinList.append(min(cnt[:, 1]))
        sqeeze = np.squeeze(cnt)
        # print(min(a[:, 0]))
        # print(max(a[:, 0]))
        # print(min(a[:, 1]))
        # print(max(a[:, 1]))

        xMin = min(sqeeze[:, 0])
        xMax = max(sqeeze[:, 0])
        yMin = min(sqeeze[:, 1])
        yMax = max(sqeeze[:, 1])

        if xMin == 0 or xMax == width - 1 or yMin == 0 or yMax == height -1:
            # print(123)
            # print(cnt)
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
            # print(xMin, xMax, yMin, yMax)
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
            # if firstIn == 
            #     pass
            # if xMin == p2[0]:
            pts1 = np.float32([p1,p2,p3,p4])
            # else:
            #     pts1 = np.float32([p2,p4,p1,p3])

            M = cv2.getPerspectiveTransform(pts1,pts2)
            rst = cv2.warpPerspective(img_org,M,(height,width))
            break
    i -= 1

cv2.imwrite('original.jpg',img_org)
cv2.imwrite('o.jpg',rst)

# height, width = img.shape[:2]
# print(width, height)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray_org = img

# i = 30
# maxArea = height * width
# while i > 0:
#     img = gray_org
#     # img = dilate(img, (i, i))
#     img = erode(img, (3, 3))
#     # img = gaussianblur(img, (3,3))
#     print(i)
#     cv2.imwrite(str(i) + 'o.jpg',img)

#     ret, thresh = cv2.threshold(img, 250, 256, cv2.THRESH_TOZERO_INV) 

#     imgEdge,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     contours.sort(key=cv2.contourArea, reverse=True)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < maxArea * 0.3:
#             continue

#         if  maxArea * 0.95 < area:
#             continue
#         epsilon = 0.005*cv2.arcLength(cnt,True)
#         approx = cv2.approxPolyDP(cnt,epsilon,True)
#         cv2.drawContours(img_org,[approx],-1,[0,255,0],2)    # 等高線の太さ
#         i = 0
#         # print(area)
#         # break
#     i-= 1
# cv2.imwrite('o.jpg',img_org)
# # グレースケールに変換
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# # キャニー法によるエッジ検出
# edges = cv2.Canny(gray,90,150,apertureSize = 3)


# kernel = np.ones((3,3),np.uint8)

# # モルフォロジー膨張変換
# edges = cv2.dilate(edges,kernel,iterations = 1)
# kernel = np.ones((5,5),np.uint8)

# # モルフォロジー収縮変換
# edges = cv2.erode(edges,kernel,iterations = 1)
# cv2.imwrite('canny.jpg',edges)

# # ハフ変換による直線検出
# lines = cv2.HoughLines(edges,1,np.pi/180,150)

# if not lines.any():
#     print('No lines were found')
#     exit()

# if filter:
#     rho_threshold = 15
#     theta_threshold = 0.1

#     # how many lines are similar to a given one
#     similar_lines = {i : [] for i in range(len(lines))}
#     for i in range(len(lines)):
#         for j in range(len(lines)):
#             if i == j:
#                 continue

#             rho_i,theta_i = lines[i][0]
#             rho_j,theta_j = lines[j][0]
#             if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
#                 similar_lines[i].append(j)

#     # ordering the INDECES of the lines by how many are similar to them
#     indices = [i for i in range(len(lines))]
#     indices.sort(key=lambda x : len(similar_lines[x]))

#     # line flags is the base for the filtering
#     line_flags = len(lines)*[True]
#     for i in range(len(lines) - 1):
#         if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
#             continue

#         for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
#             if not line_flags[indices[j]]: # and only if we have not disregarded them already
#                 continue

#             rho_i,theta_i = lines[indices[i]][0]
#             rho_j,theta_j = lines[indices[j]][0]
#             if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
#                 line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

# print('number of Hough lines:', len(lines))

# filtered_lines = []

# if filter:
#     for i in range(len(lines)): # filtering
#         if line_flags[i]:
#             filtered_lines.append(lines[i])

#     print('Number of filtered lines:', len(filtered_lines))
# else:
#     filtered_lines = lines

# for line in filtered_lines:
#     rho,theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('hough.jpg',img)
