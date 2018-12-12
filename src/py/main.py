import cv2
import numpy
import glob
import os
from imageTransformer import ImageTransformer
import urllib.request
import io


np = numpy

# ぼやかす
def gaussianblur(img: numpy.ndarray, ksize: tuple) -> numpy.ndarray:
    return cv2.GaussianBlur(img, ksize, 0)

def blur(img: numpy.ndarray, ksize: tuple) -> numpy.ndarray:
    return cv2.blur(img, ksize)

def medianBlur(img: numpy.ndarray, ksize: int) -> numpy.ndarray:
    return cv2.medianBlur(img, ksize)

def bilateralFilter(img: numpy.ndarray, d: int, sigmaColor: int, sigmaSpace: int) -> numpy.ndarray:
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

### モルフォロジー変換
def erode(img: numpy.ndarray, kernelShape: tuple) -> numpy.ndarray:
    kernel = numpy.ones(kernelShape,numpy.uint8)
    return cv2.erode(img, kernel, iterations = 1)

def dilate(img: numpy.ndarray, kernelShape: tuple) -> numpy.ndarray:
    kernel = numpy.ones(kernelShape,numpy.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

def opening(img: numpy.ndarray, kernelShape: tuple) -> numpy.ndarray:
    kernel = numpy.ones(kernelShape,numpy.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img: numpy.ndarray, kernelShape: tuple) -> numpy.ndarray:
    kernel = numpy.ones(kernelShape,numpy.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def gradient(img: numpy.ndarray, kernelShape: tuple) -> numpy.ndarray:
    kernel = numpy.ones(kernelShape,numpy.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def tophat(img: numpy.ndarray, kernelShape: tuple) -> numpy.ndarray:
    kernel = numpy.ones(kernelShape,numpy.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def blackhat(img: numpy.ndarray, kernelShape: tuple) -> numpy.ndarray:
    kernel = numpy.ones(kernelShape,numpy.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# 画像ピラミッド
def pryUp(img: numpy.ndarray) -> numpy.ndarray:
    return  cv2.pyrUp(img)


def pryDown(img: numpy.ndarray) -> numpy.ndarray:
    return  cv2.pyrDown(img)

# 色変換
def cvtColorGray(img: numpy.ndarray) -> numpy.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

datasetPath= [
    './dataset/crossword',
    './dataset/document',
    './dataset/eraser',
    './dataset/pen',
    './dataset/website',
]

compositPath= [
    './composite/crossword',
    './composite/document',
    './composite/eraser',
    './composite/pen',
    './composite/website',
]

url = "http://localhost:8080/v0.0/sudoku/generate/problem?type=img"

for i, inputDirPath in enumerate(datasetPath):
    fileList = glob.glob(inputDirPath + '/*') # type: list[str]
    for inputFilePath in fileList:
        outputFileName = 'composite_' + os.path.basename(inputFilePath).replace("_","") # type: str
        outputFilePath = compositPath[i] + '/' + outputFileName # type: str
        # print(inputFilePath)
        background = cv2.imread(inputFilePath) # type: numpy.ndarray
        # try:
        wBackground, hBackground = background.shape[:2] # type: int

        req = urllib.request.Request(url, method='POST')

        with urllib.request.urlopen(req) as res:
            body = res.read()

        with open("sudoku.jpg", "wb") as f:
            f.write(body)

        sudoku = cv2.imread("./sudoku.jpg") # type: numpy.ndarray
        # mask = sudoku.copy()
        # mask[:] = 255

                # グレースケールに変換する。
        gray = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
        # 2値化する。
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(sudoku)
        cv2.drawContours(mask, contours, -1, color=(255, 255, 255), thickness=-1)

        # theta=45, phi=45, gamma=45

        theta, phi, gamma = numpy.random.randint(-30, 30, 3)

        it = ImageTransformer(sudoku)
        sudoku = it.rotate_along_axis(theta=theta, phi=phi, gamma=gamma)

        it = ImageTransformer(mask)
        mask = it.rotate_along_axis(theta=theta, phi=phi, gamma=gamma)

        wSudoku, hSudoku = sudoku.shape[:2] # type: int

        level = [-90, 90]  # type: list[int]
        angle = np.random.randint(level[0], level[1])
        angleRad = angle/180.0*np.pi

        wSudokuRot = int(np.round(hSudoku*np.absolute(np.sin(angleRad))+wSudoku*np.absolute(np.cos(angleRad)))) # type: int
        hSudokuRot = int(np.round(hSudoku*np.absolute(np.cos(angleRad))+wSudoku*np.absolute(np.sin(angleRad)))) # type: int

        sudokuSizeRot = (wSudokuRot, hSudokuRot) # type: tuple

        scale = 1.0 # type: double
        rotationMatrix = cv2.getRotationMatrix2D((wSudoku / 2, hSudoku / 2), angle, scale) # type: numpy.ndarray

        affine_matrix = rotationMatrix.copy() # type: numpy.ndarray
        affine_matrix[0][2] = affine_matrix[0][2] -wSudoku/2 + wSudokuRot/2 # type: numpy.ndarray
        affine_matrix[1][2] = affine_matrix[1][2] -hSudoku/2 + hSudokuRot/2 # type: numpy.ndarray

        sudoku = cv2.warpAffine(sudoku, affine_matrix, sudokuSizeRot, flags=cv2.INTER_CUBIC) # type: numpy.ndarray
        mask = cv2.warpAffine(mask, affine_matrix, sudokuSizeRot, flags=cv2.INTER_CUBIC) # type: numpy.ndarray

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 左上 (x:min, y:min), 右上: (x:max, y:min, 左下: (x:min, y:max), 右下: (x:max, y:max)
        xMaxList = []
        xMinList = []
        yMaxList = []
        yMinList = []
        for k, cnt in enumerate(contours):
            cnt = np.squeeze(cnt, axis=1)
            # print("min",min(cnt[:, 0]), min(cnt[:, 1]))
            xMaxList.append(max(cnt[:, 0]))
            xMinList.append(min(cnt[:, 0]))

            yMaxList.append(max(cnt[:, 1]))
            yMinList.append(min(cnt[:, 1]))

        # print(max(xMax), min(xMin), max(yMax), min(yMin))
        xMax = max(xMaxList)
        xMin = min(xMinList)
        yMax = max(yMaxList)
        yMin = min(yMinList)

        print(xMax, xMin, yMax, yMin)
        
        sudoku = sudoku[yMin: yMax,xMin: xMax]
        mask = mask[yMin: yMax, xMin: xMax]

        tempSudoku = sudoku.copy()
        tempMask = mask.copy()

        wSudoku, hSudoku = tempSudoku.shape[:2]
        for j in range(1, 100):
            if wBackground <= wSudoku or hBackground <= hSudoku:
                if wBackground <= wSudoku:
                    wRate = (wBackground / wSudoku) * (1 - (j * 0.01))
                if hBackground <= hSudoku:
                    hRate = (hBackground / hSudoku) * (1 - (j * 0.01))

                whRate = wRate if wRate <= hRate else hRate

                whRate = whRate * np.random.randint(5, 11) * 0.1

                # print(whRate)

                tempSudoku = cv2.resize(tempSudoku, dsize=None, fx=whRate, fy=whRate)
                tempMask = cv2.resize(tempMask, dsize=None, fx=whRate, fy=whRate)

            wSudoku, hSudoku = tempSudoku.shape[:2] # type: int

            if wBackground > wSudoku and hBackground > hSudoku:
                sudoku = tempSudoku
                mask = tempMask
                xOffset = np.random.randint(0, wBackground - wSudoku + 1) # type: int
                yOffset = np.random.randint(0, hBackground - hSudoku + 1) # type: int
                roi = background[xOffset: xOffset + wSudoku, yOffset: yOffset + hSudoku] # type: numpy.ndarray
                result = np.where(mask==255, sudoku, roi) # type: numpy.ndarray
                background[xOffset: xOffset + wSudoku, yOffset: yOffset + hSudoku] = result
                # print(xOffset, wSudoku, yOffset , hSudoku)
                # ymin, ymax, xmin, xmax

                # print(yOffset, hSudoku+yOffset, xOffset, wSudoku+xOffset)

                width, height = background.shape[:2]

                xmin = str(yOffset)
                xmax = str(hSudoku+yOffset)

                ymin = str(xOffset)
                ymax = str(wSudoku+xOffset)


                outPutFormat = "_" + str(height) + "_" + str(width) + "_" + xmin + "_" + xmax + "_" + ymin + "_" + ymax + ".jpg"
                print("output: ", outputFilePath + outPutFormat)
                cv2.imwrite(outputFilePath + outPutFormat, background)
                # exit(0)
                break
            else:
                tempSudoku = sudoku.copy()
                tempMask = mask.copy()