ONNXINFERENCE = True
ONNXRENDERER = False #TF is much faster.
RENDER = True
USEWRITER = False

TRACKNUM = 3 #Number of people in video to track
ModelFolder = "./models/"
VideoFolder = "./video/"
INFERENCEVIDEO = VideoFolder + "sample.mp4" #Video file
INFERENCEVIDEOOFFSET = 0 #Skip first frames
SMPLFOLDER = "./smpl/"
#SMPLFILE = SMPLFOLDER + 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
SMPLFILE = SMPLFOLDER + 'basicModel_m_lbs_10_207_0_v1.0.0.pkl'

SYNCHRONOUSRENDERING = False
OUTPUTSCALE = 0.5


USETF = not ONNXINFERENCE or (RENDER and not ONNXRENDERER)

# if not USETF:
#     RENDER = False

if ONNXINFERENCE or ONNXRENDERER:
    import onnxruntime

if USETF:
    import tensorflow as tf


if USEWRITER:
    WRITERFPS = 24.0
    FOURCC = "DIVX" #'XVID' #'DIVX' #"avc1"
    if RENDER:
        SYNCHRONOUSRENDERING = True
        WRITEROUTPUT = VideoFolder + "output3D.mp4"
    else:
        WRITEROUTPUT = VideoFolder + "output2D.mp4"
#HALFSIZE = True #Shrink the input video to half