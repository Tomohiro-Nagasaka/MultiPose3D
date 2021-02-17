import cv2
import os    
from pathlib import Path

writer = None
def InitWriter(filename, size, FOURCC='DIVX', FPS=24.0):
    global writer
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(filename,fourcc, FPS, size)

def Write(frame):
    writer.write(frame)

def EndWriter():
    global writer
    writer.release()
    writer = None

def InitCamera(num=0):
    global cap
    cap = cv2.VideoCapture(num)
    cap.set(3,1280)
    cap.set(4,720)
    return cap

def InitVideo(filename):
    global cap
    global currentframe
    if os.path.isdir(filename):
        currentframe = 0
        cap = [str(x) for x in Path(filename).glob("**/*.png")]
    else:
        cap = cv2.VideoCapture(filename,cv2.CAP_FFMPEG)
    return cap


def GetImage():
    global cap
    global currentframe
    if isinstance(cap, list):
        if currentframe < len(cap):
            ret = True
            frame = cv2.imread(cap[currentframe])
            currentframe+=1
        else:
            ret = False
            frame = None
    else:
        ret, frame = cap.read()
    return ret, frame

def Seek(frame_no):
    global cap
    global currentframe
    if isinstance(cap, list):
        currentframe = frame_no
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)

def EndCamera():
    global cap
    if isinstance(cap, list):
        cap = None
    else:
        cap.release()
    cv2.destroyAllWindows()

def EndVideo():
    EndCamera()



