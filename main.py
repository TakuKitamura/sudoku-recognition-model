import cv2
import numpy

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

sudoku = cv2.imread('./sudoku.png') # type: numpy.ndarray
web = cv2.imread('./web.jpg') # type: numpy.ndarray

w1, h1 = sudoku.shape[:2]
w2, h2 = web.swebpe[:2]

height, width = sudoku.shape[:2]

web[0:height, 0:width] = sudoku

# cv2.imwrite('new.jpg', web)
cv2.imshow('test', web) # 画像の表示
cv2.waitKey(0)