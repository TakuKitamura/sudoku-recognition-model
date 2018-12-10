import cv2
import numpy
import glob
import os
from imageTransformer import ImageTransformer

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

for i, inputDirPath in enumerate(datasetPath):
    fileList = glob.glob(inputDirPath + '/*') # type: list[str]
    for inputFilePath in fileList:
        outputFileName = 'composite_' + os.path.basename(inputFilePath) # type: str
        outputFilePath = compositPath[i] + '/' + outputFileName # type: str
        # print(inputFilePath)
        background = cv2.imread(inputFilePath) # type: numpy.ndarray
        # try:
        wBackground, hBackground = background.shape[:2] # type: int

        sudoku = cv2.imread('./sudoku.jpg') # type: numpy.ndarray
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

        theta, phi, gamma = numpy.random.randint(-30,30,3)

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

        tempSudoku = sudoku.copy()
        tempMask = mask.copy()
        for j in range(1, 100):
            if wBackground <= wSudoku or hBackground <= hSudoku:
                if wBackground <= wSudoku:
                    wRate = (wBackground / wSudoku) * (1 - (j * 0.01))
                if hBackground <= hSudoku:
                    hRate = (hBackground / hSudoku) * (1 - (j * 0.01))

                whRate = wRate if wRate <= hRate else hRate

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
                cv2.imwrite('/Users/kitamurataku/Desktop/a/' + outputFileName, background)
                break
            else:
                tempSudoku = sudoku.copy()
                tempMask = mask.copy()