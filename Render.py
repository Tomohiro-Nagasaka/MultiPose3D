#%%
from Config import *
import numpy as np
import ModernGL
from SMPL import SMPLModel
import time


#%%


smpl0 = SMPLModel(SMPLFILE, BatchDims=1, OnlyJoints=False)
#%%
faces = smpl0.faces
verts = smpl0.v_template[0]

def Normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

norm = np.zeros( verts.shape, dtype=verts.dtype )
tris = verts[faces]            
n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
n = Normalize(n)
norm[ faces[:,0] ] += n
norm[ faces[:,1] ] += n
norm[ faces[:,2] ] += n
norm = Normalize(norm)
colors = (norm + 1) * 0.5

colors *= (0.2, 0.5, 0.3)
colors = np.sum(colors, axis=-1, keepdims=True)

from common.ImageUtil import GetColorList

ColorList = GetColorList(ModernGL.MAXPEOPLE, cmap="rainbow") #"rainbow"
colorlist = []


for i in range(ModernGL.MAXPEOPLE):
    mc = np.array(ColorList[i][:3], dtype=np.float32) * (1.0 / 255)
    mc = mc.reshape([1,1,3])
    #print(mc)
    colorlist.append(mc * (colors * 0.4 + 0.6))
#assert(False)


#%%






    
SCT = None



#%%


#%%


if ONNXRENDERER:
    import onnxruntime as rt
    import numpy as np
    onnxsess = rt.InferenceSession(ModelFolder + "RENDERMODEL2.onnx")
    print([item.name for item in onnxsess.get_inputs()])

    onnxinputs = [item.name for item in onnxsess.get_inputs()]
    onnxoutputs = [item.name for item in onnxsess.get_outputs()]
    print(onnxoutputs)

    v_template = smpl0.v_template.astype(np.float32)
    shapedirs = smpl0.shapedirs.astype(np.float32)
    J_regressor = smpl0.J_regressor.astype(np.float32)
    weights = smpl0.weights.astype(np.float32)

    def ONNXPred(img, outputs=onnxoutputs, inputs=onnxinputs):
        img = [v_template, shapedirs, J_regressor, weights] + img
        #TODO The order of inputs is changed in conversion.
        params = {inputs[len(inputs) - 1 - i]: img[i] for i in range(len(inputs))}
        #params = {inputs[3]:img[0], inputs[2]:img[1], inputs[1]:img[2], inputs[0]:img[3]}
        #params = {inputs[4]:img[0], inputs[3]:img[1], inputs[2]:img[2], inputs[1]:img[3], inputs[0]:img[4]}
        #params = {inputs[3]:img[0], inputs[2]:img[1], inputs[1]:img[2], inputs[0]:img[3], inputs[4]:img[4]}
        pred = onnxsess.run(outputs[0:1], params)
        return pred
elif False:
    import tensorflow as tf
    @tf.function
    def TFPred(inputs):
        smplverts, _, _ = rendermodel(inputs)
        return smplverts

    rendermodel = tf.saved_model.load(ModelFolder + "RENDERMODEL")   
else:
    import tensorflow as tf
    v_template = tf.constant(smpl0.v_template, dtype=tf.float32)
    shapedirs = tf.constant(smpl0.shapedirs, dtype=tf.float32)
    J_regressor = tf.constant(smpl0.J_regressor, dtype=tf.float32)
    weights = tf.constant(smpl0.weights, dtype=tf.float32)

    print(v_template.shape)
    print(shapedirs.shape)
    print(J_regressor.shape)
    print(weights.shape)

    @tf.function
    def TFPred(inputs):
        smplverts = rendermodel2([v_template, shapedirs, J_regressor, weights] + inputs)
        return smplverts

    rendermodel2 = tf.saved_model.load(ModelFolder + "RENDERMODEL2") #tf.keras.models.load_model("./RENDERMODEL2")

b = 3
rot = np.zeros(shape=(b, 24, 3, 3), dtype=np.float32)
beta = np.zeros(shape=(b,10), dtype=np.float32)
scale = np.zeros(shape=(b,1, 3), dtype=np.float32)
trans = np.zeros(shape=(b,1, 3), dtype=np.float32)
transforms = np.zeros(shape=(b,3), dtype=np.float32)

if ONNXRENDERER:
    ONNXPred([rot, beta, scale, trans])
else:
    TFPred([rot, beta, scale, trans])
#%%
ModernGL.FACES = faces
ModernGL.VERTS = verts
ModernGL.COLORLIST = colorlist
#VERTSLIST = [verts,] * ModernGL.MAXPEOPLE

if not SYNCHRONOUSRENDERING:
    ModernGL.Run()

#%%

def RenderX(image, outputs, transforms, detection0):

    _, rot, beta, scale, trans = outputs[:5]
            
    if False:
        rot = rot[:,-1]
        beta = beta[:,-1]
        scale = scale[:,-1]
        trans = trans[:,-1]

    GLscale = np.array([(2.0 / ModernGL.GLW), (2.0 / ModernGL.GLH), (2.0 / ModernGL.GLH)], dtype=np.float32).reshape([1, 3])
    s = transforms[...,2:3] * GLscale
    xyz = transforms * GLscale - 1.0
    xyz *= np.array([1, 1, -0.3], dtype=np.float32).reshape([1, 3]) #Relate scale to depth
    reverse = np.array([1, -1, 1], dtype=np.float32).reshape([1, 3])
    s *= reverse
    xyz *= reverse
    s = s[:,np.newaxis] 
    xyz = xyz[:,np.newaxis]

    scale = scale * s
    trans = trans * s + xyz


    #print(rot.shape, beta.shape, scale.shape, trans.shape)
    #print(transforms.shape)
    t1 = time.time()
    
    #beta[:,0] *= 0
    if ONNXRENDERER:
        inputs = [rot, beta, scale, trans]
        #[print(item.shape, item.dtype, type(item)) for item in inputs]
        smplverts = ONNXPred(inputs)
        smplverts = smplverts[0]
        
        #print(smplverts.shape)
    elif True:
        inputs = [rot, beta, scale, trans]
        smplverts = TFPred(inputs)
        smplverts = smplverts.numpy()
        
    
    t2 = time.time()

    N = ModernGL.MAXPEOPLE

  
    vertslist = [None,] * ModernGL.MAXPEOPLE
    detection = [False,] * ModernGL.MAXPEOPLE
    #print(detection0)
    for i, item in enumerate(detection0):

        detection[i] = item
        vertslist[i] = smplverts[i].copy()
        #print(rect)
        #img = RenderX0(smplverts[i], renderer, img, rect, scale[i,0], trans[i,0])
    ModernGL.Update(vertslist, detection, image)
    t3 = time.time()

    print("RENDERMODEL", t2 - t1)

    if SYNCHRONOUSRENDERING:
        global SCT
        if SCT is None:
            ctx = ModernGL.CreateContext()
            SCT = ModernGL.SimpleColorTriangle(OfflineCTX=ctx)

        image = SCT.offlinerender()
        return image
    
    return None

