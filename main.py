import cv2
import numpy
import glob
import os

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
        sudoku = cv2.imread('./sudoku.jpg') # type: numpy.ndarray
        # try:
        wBackground, hBackground = background.shape[:2] # type: int

        wSudoku, hsudoku = sudoku.shape[:2] # type: int

        level = [-90, 90]  # type: list[int]
        angle = np.random.randint(level[0], level[1])
        angleRad = angle/180.0*np.pi
        

        wSudokuRot = int(np.round(hsudoku*np.absolute(np.sin(angleRad))+wSudoku*np.absolute(np.cos(angleRad))))
        hSudokuRot = int(np.round(hsudoku*np.absolute(np.cos(angleRad))+wSudoku*np.absolute(np.sin(angleRad))))
        sudokuSizeRot = (wSudokuRot, hSudokuRot)

        scale = 1.0 # type: double
        rotationMatrix = cv2.getRotationMatrix2D((wSudoku / 2, hsudoku / 2), angle, scale)

        affine_matrix = rotationMatrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -wSudoku/2 + wSudokuRot/2
        affine_matrix[1][2] = affine_matrix[1][2] -hsudoku/2 + hSudokuRot/2

        sudoku = cv2.warpAffine(sudoku, affine_matrix, sudokuSizeRot, flags=cv2.INTER_CUBIC)

        # except:
        #     print("Error", inputFilePath)
        
        # print(123)
        wSudoku, hsudoku = sudoku.shape[:2]
        if wBackground > wSudoku and hBackground > hsudoku:
            print(inputFilePath, 111)
            
            xOffset = np.random.randint(0, wBackground - wSudoku + 1)
            yOffset = np.random.randint(0, hBackground - hsudoku + 1)
                        
            # xOffset = 0
            # yOffset = 0
            background[xOffset: xOffset + wSudoku, yOffset: yOffset + hsudoku] = sudoku
            cv2.imwrite('/tmp/' + outputFileName, background)

# sudoku = cv2.imread('./sudoku.png') # type: numpy.ndarray
# web = cv2.imread('./web.jpg') # type: numpy.ndarray

# w1, h1 = sudoku.shape[:2]
# w2, h2 = web.swebpe[:2]

# height, width = sudoku.shape[:2]

# web[0:height, 0:width] = sudoku

# # cv2.imwrite('new.jpg', web)
# cv2.imshow('test', web) # 画像の表示
# cv2.waitKey(0)