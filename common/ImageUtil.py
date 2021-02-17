import cv2
import os
import numpy as np

def Transform3DKP(data2D, data3D, visible):
    Mtr, rigid_mask = cv2.estimateAffinePartial2D(
                data3D[visible,:2].astype(np.float32),
                data2D[visible,:2].astype(np.float32)
            )
    if Mtr is None:
        #Could be no enough visible points
        return None

    scale = np.sqrt(np.linalg.det(Mtr[:2,:2]))

    P = np.zeros([3,4],dtype=np.float32)
    P[:2,:2] = Mtr[:2,:2]
    P[2,2] = scale
    P[:2,3] = Mtr[:2,2]
    P = P.transpose()
    
    kp3d = data3D
    N = kp3d.shape[0]
    kp3d = np.concatenate([kp3d, np.ones([N,1],dtype=np.float32)], axis=1)
    kp3d = kp3d @ P

    return kp3d

    
def cv2imread(path):
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        img = None  
    return img

def cv2imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def BatchImshow(X, rows=4, title="LARGE"):
    N = X.shape[0]
    cols = N // rows
    tiled = []
    for y in range(rows):
        images = []
        for x in range(cols):
            i = x + y * cols
            image = X[i]
            images.append(image)
        
        tiled.append(np.concatenate(images, axis=1))
    
    large = np.concatenate(tiled, axis=0)
    cv2.imshow(title, large)
    cv2.waitKey(1)


def ToSquare(rect):
    x1, y1, w, h = rect
    if w < h:
        x1 = x1 - (h - w) // 2
        w = h
    if h < w:
        y1 = y1 - (w - h) // 2
        h = w
    return x1, y1, w, h

def ToSquare2(rect):
    x1, y1, w, h = rect
    if w < h:
        y1 += (h - w) // 2
        h = w
    if h < w:
        x1 += (w - h) // 2        
        w = h
    return x1, y1, w, h

def ToSquare3(rect):
    x1, y1, w, h = rect
    size = (w + h) / 2

    x1 += (w - size) / 2
    y1 += (h - size) / 2
    h = w = size

    return x1, y1, w, h


def ToInt(rect):
    x1, y1, w, h = rect
    return int(x1), int(y1), int(w), int(h)

def pad_img_to_fit_bbox(img, x1, x2, y1, y2, REFLECT=1):

    if REFLECT == 1:
        img = np.pad(img, [(- min(0, y1), max(y2 - img.shape[0], 0)),(-min(0, x1), max(x2 - img.shape[1], 0)),(0,0)],
         'reflect')
    elif REFLECT == 0:
        img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
            -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    elif REFLECT == 2:
        img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
            -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT)  #cv2.BORDER_REPLICATE)

    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def Crop0(img, rect, REFLECT):
    x1, y1, w, h = ToInt(rect)
    x2 = x1 + w
    y2 = y1 + h
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2, REFLECT)
    return img[y1:y2, x1:x2]



def CropNoReflect(img, rect, REFLECT=0):
    return Crop0(img, rect, REFLECT)

def Crop(img, rect, REFLECT=1):
    return Crop0(img, rect, REFLECT)

def CropBorderConstant(img, rect, REFLECT=2):
    return Crop0(img, rect, REFLECT)


def Area(rect):
    x1, y1, w, h = rect
    return w * h

def AddMargin(rect, marginratio = 0.2):
    x1, y1, w, h = rect
    x2 = x1 + w
    y2 = y1 + h
    marginx = w * marginratio
    marginy = h * marginratio
    x1 -= marginx
    y1 -= marginy
    x2 += marginx
    y2 += marginy
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

def AddMargin4(rect, marginratio = [0.2,] * 4):
    x1, y1, w, h = rect
    x2 = x1 + w
    y2 = y1 + h
    # marginx = w * marginratio
    # marginy = h * marginratio
    x1 -= w * marginratio[0]
    y1 -= h * marginratio[1]
    x2 += w * marginratio[2]
    y2 += h * marginratio[3]
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def CropPoints(points, rect):
    x1, y1, w, h = rect
    points[...,0] -= x1
    points[...,1] -= y1
    return points

def NormalizePoints(points, rect):
    x1, y1, w, h = rect
    points[...,0] -= x1 + 0.5
    points[...,1] -= y1 + 0.5

    points[...,0] /= (w-1)
    points[...,1] /= (h-1)
    return points

def UnNormalizePoints(points, rect):
    x1, y1, w, h = rect
    
    points[...,0] *= (w-1)
    points[...,1] *= (h-1)
    points[...,0] += x1 + 0.5
    points[...,1] += y1 + 0.5

    return points

def NormalizePoints3D(points, rect):
    x1, y1, w, h = rect
    points[...,0] -= x1 + 0.5
    points[...,1] -= y1 + 0.5

    points[...,0] /= (w-1)
    points[...,1] /= (h-1)
    points[...,2] /= (h-1)
    return points

def UnNormalizePoints3D(points, rect):
    x1, y1, w, h = rect
    
    points[...,0] *= (w-1)
    points[...,1] *= (h-1)
    points[...,2] *= (h-1)

    points[...,0] += x1 + 0.5
    points[...,1] += y1 + 0.5

    return points


def PointsToROI(kp, Threshold=0.5, Normalized=False):

    if kp.shape[1] == 3:
        idx = kp[...,2] > Threshold
        kpX = kp[idx,:2]
    else:
        kpX = kp

    if kpX.shape[0] == 0:
        return 0, 0, 0, 0

    x1, y1 = np.amin(kpX,axis=0)
    x2, y2 = np.amax(kpX,axis=0)
    if not Normalized:
        x2 += 1
        y2 += 1
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

def PointsToROIOld(points, Threshold=0.5):
    M = points.shape[0]

    USESCORE = points.shape[1] == 3
    #print(USESCORE)
    x1 = 9999
    x2 = -9999
    y1 = 9999
    y2 = -9999
    for j in range(M):
        x = points[j,0]
        y = points[j,1]
        if USESCORE:
            score = points[j,2]
            if(score < Threshold):
                continue

        x1 = min(x, x1)
        x2 = max(x, x2)
        y1 = min(y, y1)
        y2 = max(y, y2)

    x2 += 1
    y2 += 1
    if(x1 >= x2 or y1 >= y2):
        return 0, 0, 0, 0
    else:
        w = x2 - x1
        h = y2 - y1
        return x1, y1, w, h



def SetMask(img, r = 1.0 / 8):
    W = img.shape[1]
    MARGIN = int(W * r)
    img[:,:MARGIN] = 0
    img[:,-MARGIN-1:] = 0
    return img


def Resize(img, W, H):
    return cv2.resize(img, (W, H), interpolation = cv2.INTER_LINEAR)

def ResizeNN(img, W= 256, H= 256):
    img = cv2.resize(img, (W, H), interpolation = cv2.INTER_NEAREST)
    return img

def ResizeBL(img, W= 256, H= 256):
    img = cv2.resize(img, (W, H), interpolation = cv2.INTER_LINEAR)
    return img

def ResizeImshow(title, img, W=256, H=256):
    C = img.shape[-1]
    if(len(img.shape)==3) and C > 4:
        img = np.amax(img, axis=-1)

    if img.shape[1] <= 256:
        cv2.imshow(title, ResizeNN(img, W, H))
    else:
        cv2.imshow(title, ResizeBL(img, W, H))




#%%
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

def GetColorList(maxima = 8, cmap="hsv"):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=maxima + 1, clip=True)
    mapper = cm.ScalarMappable(norm=norm,cmap=cmap)
    colors = [mapper.to_rgba(v) for v in range(maxima + 1)]
    colors = [(c[0] * 255, c[1] * 255, c[2] * 255) for c in colors]
    colors = colors[::2] + colors[1::2]
    return colors

def DrawRect(img, rect, color=(0,255,0)): 
    x, y, w, h = ToInt(rect)
    cv2.rectangle(img,(x, y),(x + w, y + h),color,3)


def DrawPoints(img, keypoints0, Normalized=False, Threshold=0.5, color=(0,255,0)):

    keypoints = keypoints0.copy()
    if Normalized:
        keypoints[...,0] = keypoints[...,0] * (img.shape[1] - 1)
        keypoints[...,1] = keypoints[...,1] * (img.shape[0] - 1)
    
    keypoints = keypoints.astype(int)[...,:2]
    for i in range(keypoints.shape[0]):
        item = keypoints[i]
        #color=(0,255,0)
        #color = np.array(mapper.to_rgba(i)) * 255
        cv2.circle(img, tuple(item), 2, color, -1)

    return img



def DrawLines(img, keypoints0, lines, Normalized=False, Threshold=0.5,color=(255,0,0), thickness=2):

    keypoints = keypoints0.copy()
    if Normalized:
        keypoints[...,0] = keypoints[...,0] * (img.shape[1] - 1)
        keypoints[...,1] = keypoints[...,1] * (img.shape[0] - 1)

    keypoints = keypoints.astype(int)[...,:2]

    for limb in lines:
        start_index, end_index = limb  # connection joint index from 0 to 16
        Joint_start = keypoints[start_index]
        Joint_end = keypoints[end_index]
        cv2.line(img, tuple(Joint_start), tuple(Joint_end),color,thickness)

def DrawLines2(img, keypoints01, keypoints02, Normalized=False, Threshold=0.5):

    keypoints = keypoints01.copy()
    if Normalized:
        keypoints[...,0] = keypoints[...,0] * (img.shape[1] - 1)
        keypoints[...,1] = keypoints[...,1] * (img.shape[0] - 1)
    
    keypoints = keypoints.astype(int)[...,:2]

    keypoints2 = keypoints02.copy()
    if Normalized:
        keypoints2[...,0] = keypoints2[...,0] * (img.shape[1] - 1)
        keypoints2[...,1] = keypoints2[...,1] * (img.shape[0] - 1)
    
    keypoints2 = keypoints2.astype(int)[...,:2]

    for i in range(keypoints.shape[0]):
        item = keypoints[i]
        item2 = keypoints2[i]
        cv2.line(img, tuple(item), tuple(item2),(0,0,255),2)

    return img


def DrawRectangle(img, keypoints0, Normalized=False, Threshold=0.5):

    keypoints = keypoints0.copy()
    if Normalized:
        keypoints[...,0] = keypoints[...,0] * (img.shape[1] - 1)
        keypoints[...,1] = keypoints[...,1] * (img.shape[0] - 1)

    keypoints = keypoints.astype(int)[...,:2]


    TL = keypoints[0]
    BR = keypoints[1]
    cv2.rectangle(img, tuple(TL), tuple(BR),(0, 0, 255),2)

def DrawPointsWithVisibility(img, keypoints0, Normalized=False, Threshold=0.5,color=(0,0,255)):

    keypoints = keypoints0.copy()
    if Normalized:
        keypoints[...,0] = keypoints[...,0] * (img.shape[1] - 1)
        keypoints[...,1] = keypoints[...,1] * (img.shape[0] - 1)

    Vis = keypoints[...,2]
    keypoints = keypoints.astype(int)[...,:2]
    

    for i in range(keypoints.shape[0]):
        if Vis[i] > Threshold:
            item = keypoints[i]
            
            #color = np.array(mapper.to_rgba(i)) * 255
            cv2.circle(img, tuple(item), 2, color, -1)

    return img


def DrawLinesWithVisibility(img, keypoints0, lines, Normalized=False, Threshold=0.5):

    keypoints = keypoints0.copy()
    if Normalized:
        keypoints[...,0] = keypoints[...,0] * (img.shape[1] - 1)
        keypoints[...,1] = keypoints[...,1] * (img.shape[0] - 1)

    Vis = keypoints[...,2]
    keypoints = keypoints.astype(int)[...,:2]
    

    for limb in lines:
        start_index, end_index = limb  # connection joint index from 0 to 16
        if Vis[start_index] > Threshold and Vis[end_index] > Threshold:
            Joint_start = keypoints[start_index]
            Joint_end = keypoints[end_index]
            cv2.line(img, tuple(Joint_start[:2]), tuple(Joint_end[:2]),(255,0,0),2)

            
                    