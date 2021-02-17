#%%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
#%%
import os

#%%
import sys
import time
import math
import Params
from Config import *

KPNUM = Params.BKPNUM
SEQLEN = Params.SEQLEN
SIZE = Params.SIZE





if USETF:
    import tensorflow as tf
    print(tf.__version__)   
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass


import random
import cv2
import numpy as np

#%%

import common.Camera as Camera
from common.ImageUtil import *
from common.ImageUtil2 import *

#%%


def VideoGenerator():
    
    mp4file = INFERENCEVIDEO
    Camera.InitVideo(mp4file)
    
    ret, img = Camera.GetImage()
    if not ret:
        print("ERROR VIDEO")
        assert(False)

    Camera.Seek(INFERENCEVIDEOOFFSET)
    while(True):
        ret, img = Camera.GetImage()
        if not ret:
            break
        #img = Preprocess(img)

        
        #print(img.shape)
        yield (img,)

gen = VideoGenerator()
data = next(gen)
Camera.Seek(0)

ImageW = data[0].shape[1]
ImageH = data[0].shape[0]


if not ONNXINFERENCE:
    dtypes = [tf.dtypes.as_dtype(item.dtype) for item in data]
    shapes = [tf.TensorShape(item.shape) for item in data]
        
    print(dtypes)
    print(shapes)
    
    def GenerateVideoDataset(arg):
        return tf.data.Dataset.from_generator(VideoGenerator, tuple(dtypes),output_shapes=tuple(shapes))

    dataset = None
    dataset = GenerateVideoDataset(None).prefetch(buffer_size=2)
    bgen = iter(dataset)
        #data = next(bgen)


    @tf.function
    def Debug0(model, image):
        return model(image, training=False)

else:
    bgen = gen


#%%


if RENDER:  
    import To3D
    import ModernGL
    from Utils import QuatToRot, RotToQuat

    ModernGL.TEXTURESIZE = 256 * 3
    ModernGL.GLW = ImageW
    ModernGL.GLH = ImageH
    
    import Render
    

    






#%%

PREIMAGE = None
PREHEATMAP = None
PRECOORD = None
PRECOORD2 = None
PREANKERS = None
PREDETECTION = None
TRACKINGID = None
TRACKCOUNT = np.zeros([TRACKNUM],dtype=np.int32)
ColorList = GetColorList(TRACKNUM)
TRACKKEYPOINTS = np.zeros([SEQLEN, TRACKNUM, KPNUM, 3],dtype=np.float32)

def CycleShift(array):
    array[:-1] = array[1:]

from scipy.optimize import linear_sum_assignment
ROT, SCALE, TRANS, BETA = None, None, None, None

def GetDistanceMatrix0(points, points2, detection, detection2, MAXDIST=0.25):

    diff = points[:,np.newaxis] - points2[np.newaxis,:]
    dmat = np.linalg.norm(diff, axis=-1)
    if len(points.shape)==3:
        dmat = np.mean(dmat, axis=-1)

    dmat[np.logical_not(detection),:] = MAXDIST
    dmat[:,np.logical_not(detection2)] = MAXDIST
    dmat = np.transpose(dmat, axes=[1,0])
    return dmat

def GetDistanceMatrix(points, points2, detection, trackcount, MAXDIST=0.25):

    detection2 = trackcount > 0
    trackmult = np.maximum((10 - trackcount) * 0.20, 1)
    #print(points.shape, detection.shape)

    diff = points[:,np.newaxis] - points2[np.newaxis,:]
    dmat = np.linalg.norm(diff, axis=-1)
    if len(points.shape)==3:
        dmat = np.mean(dmat, axis=-1)

    dmat = dmat * trackmult[np.newaxis]
    dmat[:,::TOPKNUM] *= 0.3 #XXX ALWAYS PREFER THE FIRST DETECTION

    dmat[:,np.logical_not(detection2)] = MAXDIST * 0.999
    if True:
        dmat = np.minimum(dmat, MAXDIST * (0.95 + np.random.uniform(0.01,0.04,size=dmat.shape)))
        dmat[:,::TOPKNUM] = np.minimum(dmat[:,::TOPKNUM], MAXDIST * 0.95)
    else:
        dmat = np.minimum(dmat, MAXDIST * (0.95 + np.random.uniform(0,0.04,size=dmat.shape)))
        
    dmat[np.logical_not(detection),:] = MAXDIST
    
    # print(points)
    # print(detection)
    # print(dmat)
    #dmat = np.minimum(dmat, MAXDIST)
    dmat = np.transpose(dmat, axes=[1,0])

    return dmat



if ONNXINFERENCE:
    import onnxruntime as rt
    import numpy as np
    onnxsess = rt.InferenceSession(ModelFolder + "SCN.onnx")
    print([item.name for item in onnxsess.get_inputs()])

    input_name = onnxsess.get_inputs()[0].name
    outputs = [item.name for item in onnxsess.get_outputs()]
    print(outputs)

    def ONNXPred(img, outputs=outputs, input_name=input_name):
        pred = onnxsess.run(outputs, {input_name: img})
        return pred
else:
    @tf.function
    def TFPred(image):
        return model(image, training=False)
    model = tf.saved_model.load(ModelFolder + "SCN")

#%%
COREINDICES2 = [1,5,6,8,11]
TOPKNUM = 2
TRACKNUM = 3


def Inference0(origimg):

    DEBUG = False
    DRAWANKER = not RENDER

    global PREIMAGE, PREHEATMAP, PRECOORD, PRECOORD2, PREANKERS, PREDETECTION, TRACKINGID, TRACKCOUNT, TRACKKEYPOINTS

    #TODO Use TF for affine crop
    if not isinstance(origimg, np.ndarray):
        origimg = origimg.numpy()

    Ankers = ScanningAnkers(origimg)
    random.shuffle(Ankers)
    Ankers = Ankers[:TRACKNUM]
    # print(PREDETECTION)
    if PREANKERS is not None:
        #print(PREDETECTION.shape)
        predetection = PREDETECTION #[:,0] #BATCH
        for i in range(len(PREANKERS)):
            if predetection[i] == True:
                # kp = PRECOORD2[i,0]
                # print(kp.shape)
                # ankers = KPToAnker(kp, True)
                Ankers[i] = PREANKERS[i]
    else:
        predetection = np.zeros([TRACKNUM],dtype=np.bool)
    
    IMGS, WARPS = CropByAnkers(origimg, Ankers=Ankers)
    img = np.stack(IMGS)
    warps = np.stack(WARPS)

    
    IMGSIZE = max(origimg.shape[0], origimg.shape[1])
    


    N = img.shape[0]

    imgTF = img
    
    start_time = time.time()
    if ONNXINFERENCE:
        imgTF = img.astype(np.float32) / 255
        detection, coord = ONNXPred(imgTF)
    else:
        imgTF = tf.cast(imgTF, tf.float32) / 255   
        detection, coord = TFPred(imgTF)
        detection = detection.numpy()
        coord = coord.numpy()
    print("--- %s seconds ---" % (time.time() - start_time))

    coord0 = coord.copy()
    detection0 = detection.copy()

    if True:
        for i in range(N):
            coord[i] = InvAffineTransformKP(coord[i] * (SIZE - 1), WARPS[i])

        MAXDIST = 0.30 * origimg.shape[1]
        DUPLICATEDIST = 0.04 * origimg.shape[1]
        #print(DUPLICATEDIST)




    if PRECOORD is None:
        if True:
            PRECOORD = coord[:,0]
            PREDETECTION = detection[:,0]
        
        PNUM = PRECOORD.shape[0]
        TRACKINGID = np.array(list(range(PNUM)),dtype=np.int32)

    detection = detection.reshape([-1,] + list(detection.shape)[2:])
    coord = coord.reshape([-1,] + list(coord.shape)[2:])
    

#TODO Delete duplicate.
    if True:
        distancematrix2 = GetDistanceMatrix0(coord[:,COREINDICES2], coord[:,COREINDICES2], detection, detection, MAXDIST=MAXDIST)
        for i in range(len(coord)):
            if i % TOPKNUM != 0 or TRACKCOUNT[i // TOPKNUM] <= 3:
                continue
            if not detection[i]:
                continue

            for j in range(len(coord)):
                if j % TOPKNUM == 0 and TRACKCOUNT[j // TOPKNUM] > 3:
                    continue
                if distancematrix2[i,j] < DUPLICATEDIST:
                    detection[j] = False

        
    distancematrix = GetDistanceMatrix(coord, PRECOORD, detection, TRACKCOUNT, MAXDIST=MAXDIST) #PREDETECTION[0])
        

    row_ind, col_ind = linear_sum_assignment(distancematrix)
    # print(col_ind.shape, row_ind.shape, distancematrix.shape)
    # print(col_ind, row_ind)
    #
    tracksuccess = distancematrix[row_ind, col_ind] < MAXDIST * 0.95
    # print(col_ind)
    # print(detection, TRACKCOUNT)
    # print(tracksuccess)

    
    predetectcount = np.sum(PREDETECTION)

    #print(detection)
    if True:
        detection00 = detection.copy()
        coord00 = coord.copy()
        #tracksuccess = tracksuccess[col_ind]
        detection = detection[col_ind]
        coord = coord[col_ind]
        #visibility2 = visibility2[:,col_ind]
    #print(detection)

    detectcount = np.sum(detection)

#TODO Delete duplicate.
    if True:
        distancematrix2 = GetDistanceMatrix0(coord[:,COREINDICES2], coord[:,COREINDICES2], detection, detection, MAXDIST=MAXDIST)
        for i in range(len(coord)):
            for j in range(i):
                if distancematrix2[i,j] < DUPLICATEDIST:
                    if detection[i] and detection[j]:
                        print("DUPLICATE")
                        if TRACKCOUNT[i] > TRACKCOUNT[j]:
                            detection[j] = False
                        else:
                            detection[i] = False

    detectcount2 = np.sum(detection)
    # print(detectcount, detectcount2)
    # print(detection)
    # print(tracksuccess)
    # print(col_ind // TOPKNUM)
    #DEBUG CODE
    if DEBUG:
        origimgX = origimg.copy()
        for j, item in enumerate(detection00):
            if item == True:# and TRACKCOUNT[j] >= 8:
                color = ColorList[0]

                DrawLines(origimgX, coord00[j], Params.basicskeleton, Normalized=False, color=tuple(color), thickness=4)
                DrawPoints(origimgX, coord00[j], Normalized=False)

        #cv2.imshow("ORIGIMG", origimg)
        ResizeImshow("DEBUG", origimgX, origimg.shape[1]//2, origimg.shape[0]//2)
        


    # if SINGLETRACK:
    #     detection = detection.reshape([N,-1,] + list(detection.shape)[2:])
    #     coord = coord.reshape([N,-1,] + list(coord.shape)[2:])
    #     detectionX = detection[:,0]
    #     coordX = coord[:,0]
    # else:
    #     detectionX = detection[0]
    #     coordX = coord[0]
    detectionX = detection
    coordX = coord
   
    tracksuccess = np.logical_and(tracksuccess, detectionX)

    # print(detectcount)
    # if detectcount != predetectcount:
    #     print(detectcount, predetectcount)
    #     print(scores)
    #     print(col_ind)
    #     print(TRACKINGID)
    #     print(TRACKINGID[col_ind])
    # if not REVERSE:
    #     TRACKINGID = TRACKINGID[col_ind]
    #     TRACKCOUNT = TRACKCOUNT[col_ind]

    TRACKCOUNT[np.logical_not(tracksuccess)] = np.maximum(np.sqrt(TRACKCOUNT[np.logical_not(tracksuccess)]) - 1, 0)
    #trackfailed = np.logical_not(detectionX, tracksuccess)
    #TRACKCOUNT[np.logical_not(detectionX)] = np.maximum(TRACKCOUNT[np.logical_not(detectionX)] / 2 - 1, 0)
    
    if False:
        TRACKCOUNT[np.logical_and(detectionX, np.logical_or(TRACKCOUNT < 1, col_ind % TOPKNUM == 0))] += 1
    else:
        TRACKCOUNT[detectionX] += 1

    #print(TRACKCOUNT)

    
    # if not REVERSE:
    #     TRACKKEYPOINTS = TRACKKEYPOINTS[:,col_ind]
    CycleShift(TRACKKEYPOINTS)

    # if True:
    #     visibility = detection[...,np.newaxis]

    TRACKKEYPOINTS[-1,...,:2] = coord
    TRACKKEYPOINTS[-1,...,2] = 1 #visibility2
    for j in range(TRACKNUM):
        if not detectionX[j]:
            TRACKKEYPOINTS[-1,j] = TRACKKEYPOINTS[-2,j]

        if TRACKCOUNT[j] == 1:
        #if PREDETECTION[:,col_ind] == False:
            TRACKKEYPOINTS[:,j,:,:2] = coord[j]
            TRACKKEYPOINTS[:,j,:,2] = 1 #visibility2[:,j]

    # PREIMAGE = imgTF
    # PREHEATMAP = preheatmap
    PRECOORD = coord

    
    #print(originalcoord)
    # for ankers in Ankers:
    #     DrawAnkers(origimg, ankers)

    if True:
        NewAnkers = []
        for i in range(len(detection)):
            if detection[i] == True:
                kp = coord[i]
                ankers, size = KPToAnker(kp, False)
                if size > IMGSIZE * 3.0 or size < IMGSIZE * 0.1:
                    detection[i] = False
                    ankers = None
                else:
                    if PREANKERS is not None:
                        preankers = PREANKERS[i]
                        if preankers is not None:
                            if TRACKCOUNT[i] < 3:
                                alpha = 1.0
                            else:
                                alpha = 0.6

                            ankers = ankers * alpha + preankers * (1.0 - alpha)
                            if DRAWANKER and TRACKCOUNT[i] >= 8:
                                DrawAnkers(origimg, preankers)

                NewAnkers.append(ankers)
            else:
                NewAnkers.append(None)

            
    PREANKERS = NewAnkers
    #PRECOORD2 = originalcoord
    PREDETECTION = detection


   
    imgX = img #[0]
    if not isinstance(imgX, np.ndarray):
        imgX = imgX.numpy()

    if RENDER:
        ukp = TRACKKEYPOINTS.copy() #SEQLEN, BATCH, KPNUM, 3
        ukp = np.transpose(ukp,axes=[1,0,2,3])

        imgY = None
            
        detectionY = [True,] * TRACKNUM #np.logical_and(detectionX, TRACKCOUNT >= 8)
        data, transforms, rects = To3D.GenerateDataX(imgY, ukp, detectionY)
        #print(data[0].shape)
        start_time = time.time()
        outputs = To3D.Inference(data)
        #print("VIDEOPOSE %s seconds ---" % (time.time() - start_time))

        # kp2 = outputs[0]
        # kp2 = kp2[:,-1]
        #print(kp2.shape, transforms.shape)

        _, rot, beta, scale, trans = outputs
        rot = rot[:,-1]
        beta = beta[:,-1]
        scale = scale[:,-1]
        trans = trans[:,-1]

        # rot[np.isnan(rot)]=0.0
        # trans[np.isnan(trans)]=0.0
        # scale[np.isnan(scale)]=0.5
        # beta[np.isnan(beta)]=0.0

        global ROT, BETA, SCALE, TRANS
        if ROT is None:
            ROT, BETA, SCALE, TRANS = rot, beta, scale, trans

        #print(ROT.shape, BETA.shape, SCALE.shape, TRANS.shape)
        #TODO Smoothing Function here
        #alpha = np.array([0.0,] * TRACKNUM, dtype=np.float32)
        alpha = np.array(tracksuccess, dtype=np.float32) * 0.4
        alpha4 = alpha[:,np.newaxis,np.newaxis,np.newaxis]
        alpha3 = alpha[:,np.newaxis,np.newaxis]
        alpha2 = alpha[:,np.newaxis]

        if True:
            ROT = ROT * alpha4 + rot * (1.0 - alpha4)
            BETA = BETA * alpha2 + beta * (1.0 - alpha2)
            SCALE = SCALE  * alpha3 + scale * (1.0 - alpha3)
            TRANS = TRANS * alpha3 + trans * (1.0 - alpha3)
        else:
            ROT = rot
            BETA = beta
            SCALE = scale
            TRANS = trans
        
        ROT[np.isnan(ROT)]=0.0
        TRANS[np.isnan(TRANS)]=0.0
        SCALE[np.isnan(SCALE)]=0.5
        BETA[np.isnan(BETA)]=0.0

        ROTX = QuatToRot(RotToQuat(ROT))

        outputsX = [None, ROTX, BETA, SCALE, TRANS]

        imgW = ResizeBL(origimg, ModernGL.TEXTURESIZE, ModernGL.TEXTURESIZE)
        render = Render.RenderX(imgW, outputsX, transforms, np.logical_and(detectionX, TRACKCOUNT >= 8))
            
    
    if not RENDER:

        for j, item in enumerate(np.logical_and(detectionX, TRACKCOUNT >= 8)):
            ID = TRACKINGID[j]
            

            if item == True:# and TRACKCOUNT[j] >= 8:
                color = ColorList[ID % TRACKNUM]

                DrawLines(origimg, coord[j], Params.basicskeleton, Normalized=False, color=tuple(color), thickness=4)
                DrawPoints(origimg, coord[j], Normalized=False)

        # for i in range(N):
        #     imgY = imgX[i]
        #     #img = XX[i]
        #     for j in range(coord0.shape[1]):
        #         if detection0[i, j]:
        #             ID = j
        #             color = ColorList[ID % TRACKNUM]
        #             DrawLines(imgY, coord0[i,j], Params.basicskeleton, Normalized=True, color=tuple(color))
        #             DrawPoints(imgY, coord0[i,j], Normalized=True)

        #     if not detection[i]:
        #         imgY = (imgY / 2).astype(np.uint8)
        #     cv2.imshow("IMG" + str(i), imgY)


        # ResizeImshow("ORIGIMG", origimg, origimg.shape[1]//2, origimg.shape[0]//2)

        # res = cv2.waitKey(1)
        # if res >= 0:
        #     PREANKERS = None
        #     PRECOORD = None
        #     TRACKCOUNT[:] = 0
        #     print("RESET")


    if RENDER:
        return render
    else:
        return origimg

    


def Inference():
    
    fcount = 0

    while(True):
        
        try:
            data = next(bgen)

            origimg = data[0]

            res = Inference0(origimg)
            if res is not None:
                if not RENDER or SYNCHRONOUSRENDERING:
                    ResizeImshow("RES", res, int(res.shape[1] * OUTPUTSCALE), int(res.shape[0] * OUTPUTSCALE))
                    cv2.waitKey(1)

                if USEWRITER:                  
                    if Camera.writer is None:
                        print("Init Writer")
                        Camera.InitWriter(WRITEROUTPUT, (res.shape[1], res.shape[0]), FOURCC=FOURCC)
                    print("Writing")
                    Camera.Write(res)

        except Exception as e:
            print(e)
            break

        print(fcount)
        fcount += 1

    

Inference()



