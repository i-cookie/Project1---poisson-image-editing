import cv2
from GUI_helpers import genMask
from GUI_helpers import setRectPos
from Possion_processor import Pos_processor

def inputData():
    print('Reading image...')
    global srcImg, tgtImg
    srcImg = cv2.imread('D:\\CodeHome\\CG\\Project1 - possion image editing\\Components\\src.png')
    tgtImg = cv2.imread('D:\\CodeHome\\CG\\Project1 - possion image editing\\Components\\tgt.png')

def generateMask():
    print('Generating mask...')
    global srcImg, mask, minX, minY
    mask, minX, minY = genMask(srcImg)

def paste():
    print('Setting Pasting Position...')
    global tgtImg, pastePosition
    pastePosition = setRectPos(tgtImg)

def PossionEdit():
    print('Generating Image...')
    global srcImg, tgtImg, mask, minX, minY, pastePosition
    app = Pos_processor(srcImg, tgtImg, mask, minX, minY, pastePosition)
    res = app.run()
    cv2.imshow('result', res)
    cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    cv2.waitKey(0)
    cv2.imwrite('result.png', res)
    print('Done.')

def main():
    inputData()
    generateMask()
    paste()
    PossionEdit()

if __name__ == "__main__":
    main()