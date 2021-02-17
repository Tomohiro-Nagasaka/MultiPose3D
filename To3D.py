#%%
from Config import *
from Params import *
from common.ImageUtil import *
from common.ImageUtil2 import *
import cv2


W = H = 256


def PreProcess(ukp, img, Threshold=0.1, isSPIN = True):
    
    if isSPIN:
        kpX = ukp * KPMASK[:,np.newaxis]
    else:
        kpX = ukp * KPMASK[BasicIndex][:,np.newaxis]

    box = PointsToROI(kpX, Threshold=Threshold)
    box = AddMargin4(box, np.float32([0.5, 0.5, 0.5, 0.60]))
    #box = AddMargin(box, 0.5)
    box = ToSquare(box)

    #kp = NormalizePoints(ukp.copy(),box)
    kp = NormalizePoints(ukp, box)

    invisible = ukp[:,2] < Threshold
    visible = np.logical_not(invisible)

    #kp = np.concatenate([kp[:,:2], ukp[:,2:3]], axis=1)

    kp[invisible,:DIMENSION-1]=INFINITY
    kp[invisible,DIMENSION-1]=0

    if img is not None:
        img = CropBorderConstant(img, box)
        img = cv2.resize(img,(W,H))

    scale = 1.0 / box[2]
    x, y, _, _ = box
    s = 1.0 / scale

    return kp, img, (x, y, s), box

def GenerateDataX(img0, ukp0, detection):
    THRESH = 0.01
    
    #ukp SEQLEN, TRACKNUM, KPNUM, 3
    BATCH = np.sum(detection)

    if img0 is not None:
        newimg = np.zeros([BATCH, H, W, 3], dtype=np.uint8)
    else:
        newimg = None

    points = np.ones([BATCH, SEQLEN, ukp0.shape[2], DIMENSION], dtype=np.float32) * INFINITY
    transforms = np.zeros([BATCH, 3], dtype=np.float32)
#     params = np.zeros([BATCH, SEQLEN, APARAMNUM], dtype=np.float32)
#     flags = np.zeros([BATCH, SEQLEN, FLAGSNUM], dtype=np.int32)

    rects = []

    j = 0
    for k, item in enumerate(detection):
        if item == True:
            ukp = ukp0[k]

            _, img, transform, rect = PreProcess(ukp[-1].copy(), img0, isSPIN=False)
            kp = NormalizePoints(ukp, rect)
            for i in range(SEQLEN):       
                invisible = ukp[i,:,2] < THRESH
                #visible = np.logical_not(invisible)
                kp[i,invisible,:DIMENSION-1]=INFINITY
                kp[i,invisible,DIMENSION-1]=0

            #print(kp)

            points[j, :] = kp
            if img0 is not None:
                newimg[j] = img
            transforms[j] = transform
            rects.append(rect)
            j += 1


    if newimg is not None:
        newimg = newimg.astype(np.float32) / 255
    basicpoints = points #[:,:,BasicIndex]
    #basicpoints = basicpoints[...,[0,1,3]]

    kpZ = basicpoints[:,-1,:,:2]
    kpZ = np.clip(kpZ, 0.0, 1.0) #.astype(np.int32)

    #print(kpZ.shape, kpZ.dtype)
    #print(basicpoints.shape)
    #print(basicpoints[:,0])
    return [newimg, points, basicpoints, kpZ], transforms, rects



if ONNXINFERENCE:
    import onnxruntime as rt
    import numpy as np
    onnxsess = rt.InferenceSession(ModelFolder + "VIDEOPOSE.onnx")
    print(onnxsess.get_providers())
    # onnxsess.set_providers(['CPUExecutionProvider'])
    # print(onnxsess.get_providers())
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
    model = tf.saved_model.load(ModelFolder + "VIDEOPOSE")
        


def Inference(data):
    newimg, points, basicpoints, kpZ = data

    inputs = [basicpoints]


    if ONNXINFERENCE:
        output = ONNXPred(basicpoints)
        #[print(item.shape) for item in output]
        if False:
            beta, rot, scale, points, trans = output
        else:
            beta, rot, scale, trans = output
        output = [None, rot, beta, scale, trans]

    else:
        output = TFPred(basicpoints)
        #output = FeatureToPointsSequence(output, SEQLEN)
        if isinstance(output, list):
            output = [item.numpy() for item in output]
            output = [None,] + output
        else:
            output = output.numpy()

    return output
