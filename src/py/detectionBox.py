import tensorflow as tf
import cv2
import numpy as np
import math
np.set_printoptions(threshold=np.inf)
def dilate(img: np.ndarray, kernelShape: tuple) -> np.ndarray:
    kernel = np.ones(kernelShape,np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

def gaussianblur(img: np.ndarray, ksize: tuple) -> np.ndarray:
    return cv2.GaussianBlur(img, ksize, 0)

def closing(img: np.ndarray, kernelShape: tuple) -> np.ndarray:
    kernel = np.ones(kernelShape,np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def opening(img: np.ndarray, kernelShape: tuple) -> np.ndarray:
    kernel = np.ones(kernelShape,np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    step = "2913"
    with tf.gfile.GFile("/Users/kitamurataku/work/sudoku-recognition-model/tf/models/model/frozenModel/sudoku-" + step + "/frozen_inference_graph.pb", 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

sampleImg = cv2.imread("./c.jpg")

originalImg = sampleImg.copy()

height, width = sampleImg.shape[:2]
# print(height, width)

sampleImg= cv2.resize(sampleImg,dsize=(299,299), interpolation = cv2.INTER_CUBIC)

np_image_data = np.asarray(sampleImg)

np_final = np.expand_dims(np_image_data, axis=0)

(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: np_final})

boxes = np.squeeze(boxes)
scores = np.squeeze(scores)
# print(scores)
max_boxes_to_draw = boxes.shape[0]
min_score_thresh = 0.01
for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
        box = tuple(boxes[i].tolist())
        # print(box)
        spaceRate = 0.05
        ymin = (box[0]*height) - (height * spaceRate) if (box[0]*height) - (height * spaceRate) > 0 else (box[0]*height)
        xmin = (box[1]*width) - (width * spaceRate) if (box[1]*width) - (width * spaceRate) > 0 else (box[1]*width)
        ymax = (box[2]*height) + (height * spaceRate) if (box[2]*height) + (height * spaceRate) > 0 else (box[2]*height)
        xmax = (box[3]*width) + (width * spaceRate) if (box[3]*width) + (width * spaceRate) > 0 else (box[3]*width)
        print(ymin, xmin, ymax, xmax)
        cropImg = originalImg[math.ceil(ymin):math.ceil(ymax), math.ceil(xmin):math.ceil(xmax)]
        print(77)
        cv2.imwrite('crop.jpg',cropImg)

        img_org = cropImg

        img = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)

        gray_org_img = img

        height, width = img.shape[:2]

        if height > 10000 or width > 10000:
            exit(1)

        j = 50
        m = 0
        while j > 0:
            print(j)
            img = gray_org_img
            green = img_org.copy()
            img = opening(img, (j, j))

            blur = gaussianblur(img,(3,3))

            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)

            imgEdge,contours,hierarchy = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # lsd = cv2.createLineSegmentDetector(0)

            # lines, _ = lsd.detect(th3)[:2]

            # # print(w)

            # # th3 = lsd.drawSegments(th3,lines)

            # for i in range(len(lines)):
            #     for x1,y1,x2,y2 in lines[i] :
            #         cv2.line(th3,(x1,y1),(x2,y2),(0,0,0),10)

            cv2.imwrite('mono_' + str(j) + '.jpg',th3)

            contours.sort(key=cv2.contourArea, reverse=True)

            contours = contours[:5]

            maxArea = height * width
            for cnt in contours:
                m+=1
                sqeeze = np.squeeze(cnt)

                # xMin = min(sqeeze[:, 0])
                # xMax = max(sqeeze[:, 0])
                # yMin = min(sqeeze[:, 1])
                # yMax = max(sqeeze[:, 1])
                # [[229 182]
                # [230 181]
                # [939 181]
                # [940 182]
                # [940 891]
                # [939 892]
                # [230 892]
                # [229 891]]
                xMinIndex = np.argmin(sqeeze[:, 0])
                p1 = [sqeeze[:, 0][xMinIndex], sqeeze[:, 1][xMinIndex]]

                xMaxIndex = np.argmax(sqeeze[:, 0])
                p2 = [sqeeze[:, 0][xMaxIndex], sqeeze[:, 1][xMaxIndex]]

                yMinIndex = np.argmin(sqeeze[:, 1])
                p3 = [sqeeze[:, 0][yMinIndex], sqeeze[:, 1][yMinIndex]]

                yMaxIndex = np.argmax(sqeeze[:, 1])
                p4 = [sqeeze[:, 0][yMaxIndex], sqeeze[:, 1][yMaxIndex]]

                if p1[0] == 0 or p2[0] == width - 1 or p3[1] == 0 or p4[1] == height -1:
                    continue

                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt,True)
                k = cv2.isContourConvex(cnt)

                epsilon = 0.00001 * perimeter
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                cv2.drawContours(green,[approx],-1,[0,255,0],2)
                cv2.imwrite('green_' + str(m) + '_' + str(j) + '.jpg', green)
                if area >= maxArea/3 and area <= maxArea:

                    print(sqeeze)
                    # epsilon = 0.01 * perimeter
                    # approx = cv2.approxPolyDP(cnt,epsilon,True)
                    # cv2.drawContours(img_org,[approx],-1,[0,255,0],2)
                    # j = 0
                    # p1=[-1, -1]
                    # p2=[-1, -1]
                    # p3=[-1, -1]
                    # p4=[-1, -1]
                    # firstIn = -1
                    # for v in np.squeeze(cnt):
                    #     # print(v)
                    #     if v[0] == xMin:
                    #         p1[0], p1[1] = v
                    #         firstIn = 1
                    #         print(1)
                    #     elif v[1] == yMin:
                    #         p2[0], p2[1] = v
                    #         print(2)
                    #     elif v[1] == yMax:
                    #         p3[0], p3[1] = v
                    #         print(3)
                    #     elif v[0] == xMax:
                    #         p4[0], p4[1] = v
                    #         firstIn = 4
                    #         print(4)
                    print(p1, p2, p3, p4)
                    pts2 = np.float32([[0,0],[height,0],[0,width],[height,width]])
                    if p1[1] < p4[1]:
                        pts1 = np.float32([p1,p2,p3,p4])
                    else:
                        pts1 = np.float32([p2,p4,p1,p3])
                    M = cv2.getPerspectiveTransform(pts1,pts2)

                    rst = cv2.warpPerspective(img_org,M,(height,width))
                    minLength = height if height > width else width
                    rst = cv2.resize(rst,(minLength,minLength))
                    # cv2.imwrite('original.jpg',img_org)
                    cv2.imwrite('result.jpg',rst)
                    exit(0)
                    break
            j -= 1
    exit(0)
