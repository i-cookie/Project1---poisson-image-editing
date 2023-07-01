import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import operator as op

class Pos_processor:

    src = None
    destination = None
    mask = None
    srcPos = [0, 0]
    offset = [0, 0]
    boundary_buffer = list()
    interior_buffer = list()

    def __init__(self, src, destination, mask, sx, sy, paste):
        self.src = src
        self.destination = destination
        self.mask = mask
        self.srcPos = [sy, sx]
        self.offset = [paste[0], paste[1]]

    def fillBuffer(self):
        height = self.mask.shape[0]
        width = self.mask.shape[1]
        neighbors = [[-1, 0, 0, 1], [0, -1, 1, 0]]
        for i in range(height):
            for j in range(width):
                if self.mask[i, j] == 0:
                    flag = False
                    for k in range(4):
                        ni = i+neighbors[0][k]
                        nj = j+neighbors[1][k]
                        if ni >= 0 and ni < height and nj >= 0 and nj < width and \
                            self.mask[ni, nj] == 255:
                                flag = True
                                break
                    if flag:
                        self.boundary_buffer.append([i, j])
                elif self.mask[i, j] == 255:
                    self.interior_buffer.append([i, j])
                else:
                    print('mask error!')
                    return

    def doOneChannel(self, src, dst):
        ptr = [0]
        ind = list()
        data = list()
        Rhs = list()
        neighbors = [[-1, 0, 0, 1], [0, -1, 1, 0]]
        rear = 0
        for i in range(len(self.interior_buffer)):
            Rhs.append(0)
            y = self.interior_buffer[i][0]
            x = self.interior_buffer[i][1]
            Np = 0
            for k in range(4):
                ny = y + neighbors[0][k]
                nx = x + neighbors[1][k]
                
                if ny >= 0 and ny < self.mask.shape[0] and nx >= 0 and nx < self.mask.shape[1]:
                    Np += 1
                    Vpq = int(src[y + self.srcPos[0], x + self.srcPos[1]]) - int(src[ny + self.srcPos[0], nx + self.srcPos[1]])
                    Rhs[i] += Vpq
                    if self.mask[ny, nx] == 255:
                        rear += 1
                        ind.append(self.interior_buffer.index([ny, nx]))
                        data.append(-1)
                    elif [ny, nx] in self.boundary_buffer:
                        Rhs[i] += dst[ny + self.offset[0], nx + self.offset[1]]
            rear += 1
            ptr.append(rear)
            ind.append(i)
            data.append(Np)
        Rhs = np.array(Rhs, dtype=int)
        sp_A = csr_matrix((data, ind, ptr), shape = (len(self.interior_buffer), len(self.interior_buffer)), dtype=int)
        solution = spsolve(sp_A, Rhs)
        np.set_printoptions(threshold=10000)
        solution = np.clip(solution, 0, 255)
        # print(solution)
        for i in range(len(self.interior_buffer)):
            y = self.interior_buffer[i][0] + self.offset[0]
            x = self.interior_buffer[i][1] + self.offset[1]
            dst[y, x] = solution[i]
        return dst

    def run(self):
        self.fillBuffer()
        bSrc, gSrc, rSrc = cv2.split(self.src)
        bDst, gDst, rDst = cv2.split(self.destination)
        print('doing B channel')
        bImg = self.doOneChannel(bSrc, bDst)
        # cv2.imshow('B', bImg)
        print('doing G channel')
        gImg = self.doOneChannel(gSrc, gDst)
        # cv2.imshow('G', gImg)
        print('doing R channel')
        rImg = self.doOneChannel(rSrc, rDst)
        # cv2.imshow('R', rImg)
        return cv2.merge([bImg, gImg, rImg])
