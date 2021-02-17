#%%
import cv2
import math
import random
import numpy as np
from common.ImageUtil import *


KPMASKBASIC = np.float32([
    0,#'headtop',        # 0
    1,#'OP Neck',        # 1
    1,#'OP RShoulder',   # 2
    1,#'OP RElbow',      # 3
    0,#'OP RWrist',      # 4
    1,#'OP LShoulder',   # 5
    1,#'OP LElbow',      # 6
    0,#'OP LWrist',      # 7
    1,#'OP RHip',        # 8
    1,#'OP RKnee',       # 9
    0,#'OP RAnkle',      # 10
    1,#'OP LHip',        # 11
    1,#'OP LKnee',       # 12
    0,#'OP LAnkle',      # 13
    0,#'OP LBigToe',     # 14
    0,#'OP RBigToe',     # 15
    0,#'OP Nose',        # 16
    0,#'OP REar',        # 17
    0,#'OP LEar',        # 18  
 
    ])

KPMASK = np.float32(
[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,])

def TransformBoxReverse(box, param):
    x, y, w, h = box
    xx, yy, s = param
    x -= xx
    y -= yy
    
    return x/s, y/s, w/s, h/s

def TransformKP(kp, param):
    x, y, s = param
    kpX = kp.copy()
    kpX[:,:2] *= s
    kpX[:,0] += x
    kpX[:,1] += y
    return kpX

def TransformKPReverse(kp, param):
    x, y, s = param
    kpX = kp.copy()
    kpX[:,0] -= x
    kpX[:,1] -= y
    kpX[:,:2] /= s
    return kpX

def MidPoint(kp1, kp2):
    if len(kp1) ==2:
        return (kp1 + kp2) / 2

    v1 = kp1[2]
    v2 = kp2[2]
    #return kp1[:2]
    return (kp1[:2] * v1 + kp2[:2] * v2) / (v1 + v2)


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
    return result

def AffineTransformKP(kp0, mat):
    kp = kp0.copy()

    if(mat.shape[1] == 2):
        kp[...,:2] = kp[...,:2] @ mat.T
    elif(mat.shape[1] == 3):
        kp[...,:2] = kp[...,:2] @ mat[:,:2].T + mat[:,2]
    else:
        assert(False)
    return kp

def InvAffineTransformKP(kp0, mat):
    kp = kp0.copy()

    if(mat.shape[1] == 2):
        kp[...,:2] = kp[...,:2] @ np.linalg.inv(mat.T)
    elif(mat.shape[1] == 3):
        kp[...,:2] = (kp[...,:2] - mat[:,2]) @ np.linalg.inv(mat[:,:2].T)
    else:
        assert(False)
    return kp

def rotateKP(kp0, angle, center=(0.0, 0.0)):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat = rot_mat[...,:2].astype(np.float32)
    kp = AffineTransformKP(kp0, rot_mat)
    return kp, rot_mat

def DrawAnkers(img, ankers0):
    ankers = ankers0.astype(np.int32)
    for i in range(len(ankers)):
        i2 = i - 1
        p1 = ankers[i]
        p2 = ankers[i2]
        cv2.line(img, tuple(p1), tuple(p2), (0, 255,0), 2)

def SetCenter(rect, center):
    x0,y0=center
    x1,y1,w,h=rect
    x2=x1+w
    y2=y1+h
    
    dx1 = x0 - x1
    dx2 = x2 - x0
    if dx1 > dx2:
        w = dx1 * 2
        #x2 = x0 + dx1
    else:
        w = dx2 * 2
        x1 = x0 - dx2

    dy1 = y0 - y1
    dy2 = y2 - y0
    if dy1 > dy2:
        h = dy1 * 2
        #y2 = y0 + dy1
    else:
        h = dy2 * 2
        y1 = y0 - dy2

    return x1,y1,w,h


def ToAnkers(rect):
    x,y,w,h=rect
    ankers = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],dtype=np.float32)
    return ankers


def AffineCrop(src, srcTri, dstTri, w=256, h=256):
    warp_mat = cv2.getAffineTransform(srcTri[:3], dstTri[:3])
    warp_dst = cv2.warpAffine(src, warp_mat, (w, h))
    return warp_dst, warp_mat


def GetRandomRect(W, H, SIZE):
    

    if W < SIZE:
        x = -np.random.uniform(0, SIZE - W)
    else:
        x = np.random.uniform(0, W - SIZE)

    if H < SIZE:
        y = -np.random.uniform(0, SIZE - H)
    else:
        y = np.random.uniform(0, H - SIZE)

    return x, y, SIZE, SIZE

def RandomOcclusion(img, ratio=0.3):
    if np.random.uniform() < ratio:
        return

    position = np.random.uniform()
    W = img.shape[1]
    H = img.shape[0]
    ratio = 0.8
    if np.random.uniform() < 0.5:
        
        if position < 0.5:
            P = int(W * position * ratio)
            img[:,:P] = 0
        else:
            P = int(W * (1.0 -position) * ratio)
            img[:,-P:] = 0

    else:
        if position < 0.35:
            P = int(H * position * ratio)
            img[:P] = 0
        else:
            P = int(H * (1.0 -position) * ratio)
            img[-P:] = 0

def ScanningAnkersX(img):
    W = img.shape[1]
    H = img.shape[0]

    #SIZE = min(W, H)
    SIZE = (W + H) / 2
    Ankers = []
    Ankers.append(ToAnkers(GetRandomRect(W, W, SIZE)))
    Ankers.append(ToAnkers(GetRandomRect(W, H, SIZE * 0.7)))
    Ankers.append(ToAnkers(GetRandomRect(W, H, SIZE * 0.5)))
    Ankers.append(ToAnkers(GetRandomRect(W, H, SIZE * 0.4)))

    return Ankers


def ScanningAnkers(img):
    W = img.shape[1]
    H = img.shape[0]

    assert(W > H)
    Ankers = []

    margin = np.random.uniform(0, 0.20)

    rect = 0, 0, W, W
    rect = AddMargin(rect, -margin)
    Ankers.append(ToAnkers(rect))

    rect = 0, 0, H, H
    rect = AddMargin(rect, -margin)
    Ankers.append(ToAnkers(rect))

    rect = W - H, 0, H, H
    rect = AddMargin(rect, -margin)
    Ankers.append(ToAnkers(rect))

    rect = (W - H) / 2, 0, H, H
    rect = AddMargin(rect, -margin)
    Ankers.append(ToAnkers(rect))

    return Ankers

def CropByAnkers(img, Ankers=None):
    BaseAnkers = ToAnkers((0,0,256,256))

    if Ankers is None:
        Ankers = ScanningAnkers(img)

    IMGS = []
    WARPMAT = []

    for ankers in Ankers:
        crop, warp_mat = AffineCrop(img, ankers, BaseAnkers)
        #warpedkp = AffineTransformKP(kpsX, warp_mat)

        IMGS.append(crop)
        WARPMAT.append(warp_mat)

    return IMGS, WARPMAT


def BoxAugment(box, m=0.12):

    rr = np.random.normal(0, m, 1)
    rr = np.minimum(rr, 0.25)
    rr = np.maximum(rr, -0.25)

    s = 0.8
    rr2 = np.random.normal(0, m * s, 1)
    rr2 = np.minimum(rr2, 0.25 * s)
    rr2 = np.maximum(rr2, -0.25 * s)

    s = 0.8
    rr3 = np.random.normal(0, m * s, 1)
    rr3 = np.minimum(rr3, 0.25 * s)
    rr3 = np.maximum(rr3, -0.25 * s)


    a = rr[0]
    b = rr2[0]
    c = rr3[0]

    boxmargin = np.float32([a+c, a+b, a-c, a-b])
    newbox = AddMargin4(box, boxmargin)
    return newbox

def GenerateRandomCrops(imgX, kpsX, SIZEX, debug=False):
    BaseAnkers = ToAnkers((0,0,256,256))

    INDICES = []
    IMGS = []
    KPS = []
    SIZE = []
    WARPMAT = []
    
    if np.random.uniform() < 0.05:

        warp_mat = cv2.getAffineTransform(BaseAnkers[:3], BaseAnkers[:3])
        crop = np.zeros([256,256,3],dtype=np.uint8)
        warpedkp = kpsX * 0
        size = 256
        INDICES.append(-1)
        IMGS.append(crop)
        KPS.append(warpedkp)
        SIZE.append(size)
        WARPMAT.append(warp_mat)

    W = imgX.shape[1]
    H = imgX.shape[0]

    for i in range(len(SIZEX)):
        if np.random.uniform() < 0.25:
            size = SIZEX[i]
            angle = np.random.normal(scale=7.0)
            

            rect = GetRandomRect(W, H, size)
            rect = BoxAugment(rect, m=0.03)
            #rect = ToSquare(rect)

            # rect = ToSquare(rect)
            x,y,w,h = rect
            center = (x+w/2, y+h/2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rot_mat = rot_mat[...,:2].astype(np.float32)

            ankers = ToAnkers(rect)
            ankers = ankers @ rot_mat
            #print(ankers)

            if debug:
                DrawAnkers(imgX, ankers)

            crop, warp_mat = AffineCrop(imgX, ankers, BaseAnkers)
            warpedkp = AffineTransformKP(kpsX, warp_mat)

            if debug:
                DrawPointsWithVisibility(crop, warpedkp.reshape([-1, 3]))
            
                print(warp_mat)
                cv2.imshow("CROP", crop)
                cv2.waitKey(1000)
                print(rect)

            INDICES.append(-1)
            IMGS.append(crop)

            #print(warpedkp.shape)
            KPS.append(warpedkp)
            SIZE.append(size)
            WARPMAT.append(warp_mat)

    if debug:
        cv2.imshow("IMG", imgX)
        cv2.waitKey(1)


    return INDICES, IMGS, KPS, SIZE, WARPMAT


def KPToAnker(kp, augment=False):
    assert(len(kp.shape)==2)

    p1 = MidPoint(kp[2],kp[5])
    p2 = MidPoint(kp[8],kp[11])

    # if debug:
    #     cv2.line(imgX, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), (0,255,0))

    orientation = p2 - p1
    if(orientation[0] == 0 and orientation[1] == 0):
        angle = 0
    else:
        angle = math.atan2(orientation[1], orientation[0])
        angle *= (180 / math.pi)
        angle -= 90

        if augment:
            angle += np.random.normal(scale=7.0)

    #print(angle)
    
    kpX, rot_mat = rotateKP(kp, angle)
    #rect = PointsToROI(kp)
    # print(rect)
    if kpX.shape[-1] == 3:
        kpX[...,2] = kpX[...,2] * KPMASKBASIC
    else:
        kpX = np.concatenate([kpX, KPMASKBASIC[:,np.newaxis]], axis=-1)

    rect = PointsToROI(kpX)
    rect = SetCenter(rect, p2 @ rot_mat.T)
    
    rect = ToSquare(rect)
    rect = AddMargin(rect, 0.6)
    size = rect[2]

    if augment:
        rect = BoxAugment(rect, m=0.03)
    #rect = ToSquare(rect)
    
    ankers = ToAnkers(rect)
    ankers = ankers @ rot_mat
    return ankers, size


def GenerateCrops(imgX, kpsX, debug=False):
    BaseAnkers = ToAnkers((0,0,256,256))

    INDICES = []
    IMGS = []
    KPS = []
    SIZE = []
    WARPMAT = []

    BODYIDX = [2,5,8,11]
    visibility = np.sum(kpsX[:,BODYIDX,2], axis=1) >=3
    for i in range(len(visibility)):
        if visibility[i]:
            kp = kpsX[i]
            
            ankers, size = KPToAnker(kp, True)
            if debug:
                DrawAnkers(imgX, ankers)

            crop, warp_mat = AffineCrop(imgX, ankers, BaseAnkers)
            warpedkp = AffineTransformKP(kpsX, warp_mat)

            if debug:
                DrawPointsWithVisibility(crop, warpedkp.reshape([-1, 3]))
            
                print(warp_mat)
                cv2.imshow("CROP", crop)
                cv2.waitKey(1000)
                print(rect)

            INDICES.append(i)
            IMGS.append(crop)
            KPS.append(warpedkp)
            SIZE.append(size)
            WARPMAT.append(warp_mat)

    if debug:
        cv2.imshow("IMG", imgX)
        cv2.waitKey(1)

        print(visibility)

    return INDICES, IMGS, KPS, SIZE, WARPMAT