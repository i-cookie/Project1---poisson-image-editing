import cv2
import numpy as np

polygon = list()
Image = None
canvas = None
lButtonDown = False
# helper variables for clipping the mask
minX = -1
maxX = -1
minY = -1
maxY = -1
# top left corner of the pasting position
tlc = [-1, -1]

def customROI(event, x, y, flags, param):
    global polygon, Image, canvas, lButtonDown, minX, minY, maxX, maxY
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 1 or y < 1 or x > canvas.shape[1]-2 or y > canvas.shape[0]-2:
            return
        lButtonDown = True
        polygon = list()
        polygon.append([x, y])
        minX = maxX = x
        minY = maxY = y

    if event == cv2.EVENT_MOUSEMOVE:
        if lButtonDown == False or polygon[0][0] < 1 or polygon[0][1] < 1 or polygon[0][0] > canvas.shape[1]-2 or polygon[0][1] > canvas.shape[0]-2:
            return
        polygon.append([x, y])

        minX = min(minX, x)
        maxX = max(maxX, x)
        minY = min(minY, y)
        maxY = max(maxY, y)

        canvas[:, :, :] = Image[:, :, :]
        cv2.polylines(canvas, [np.array(polygon, dtype = np.int32)], True, (255, 0, 0), 2, 8, 0)

    if event == cv2.EVENT_LBUTTONUP:
        lButtonDown = False
        polygon.append([x, y])

        minX = min(minX, x)
        maxX = max(maxX, x)
        minY = min(minY, y)
        maxY = max(maxY, y)

        canvas[:, :, :] = Image[:, :, :]
        cv2.polylines(canvas, [np.array(polygon, dtype = np.int32)], True, (255, 0, 0), 2, 8, 0)

def genMask(img):
    global canvas, Image, polygon, minX, minY, maxX, maxY
    canvas = img
    Image = np.copy(img)
    cv2.namedWindow('Select ROI', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Select ROI', customROI)

    while True:
        cv2.imshow('Select ROI', canvas)
        terminate = cv2.waitKey(1)
        if terminate == 32 and minX != -1:
            break
    cv2.destroyAllWindows()

    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype = np.int32)], (255, 255, 255))

    mask = mask[minY-1: maxY+2, minX-1: maxX+2]
    canvas[:, :, :] = Image[:, :, :]
    cv2.imwrite('mask.png', mask)

    return mask, minX, minY

def setPos(event, x, y, flags, param):
    global Image, canvas, tlc, minX, minY
    if event == cv2.EVENT_LBUTTONDOWN:
        if x + (maxX - minX) > Image.shape[1] or y + (maxY - minY) > Image.shape[0]:
            return
        tlc = [y, x]
        canvas[:, :, :] = Image[:, :, :]
        ptr = [[x, y], [x + (maxX-minX), y], [x + (maxX-minX), y + (maxY-minY)], [x, y + (maxY-minY)]]
        cv2.polylines(canvas, [np.array(ptr, dtype = np.int32)], True, (255, 0, 0), 2, 8, 0)

def setRectPos(img):
    global Image, canvas, tlc
    canvas = img
    Image = np.copy(img)
    cv2.namedWindow('Select Pasting Position', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Select Pasting Position', setPos)

    while True:
        cv2.imshow('Select Pasting Position', canvas)
        terminate = cv2.waitKey(1)
        if terminate == 32 and tlc[0] != -1:
            break
    cv2.destroyAllWindows()
    canvas[:, :, :] = Image[:, :, :]

    return tlc