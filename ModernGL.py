#%%
from Config import *

import moderngl
import moderngl_window as mglw

import os
import sys
import numpy as np
import time

def CreateContext():
    return moderngl.create_standalone_context()

MAXPEOPLE = 8
TEXTURESIZE = 512
GLW = 512
GLH = 512

class Example(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ModernGL Example"
    
    window_size = (1280, 720)
    aspect_ratio = 1280 / 720

    resizable = True
    samples = 4

    resource_dir = os.path.normpath("./") #os.path.normpath(os.path.join(__file__, '../../data'))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)
        os._exit(1) #Terminate python from thread
        


#%%
from array import array
def TextureShader(ctx):
    
    # 頂点座標 (x,y,z)、テクスチャ座標 (x,y)
    vbo = ctx.buffer(array('f',
    [
        # pos xyz    uv
        -1.0,  1.0, 0.0, 1.0,
        -1.0, -1.0, 0.0, 0.0,
         1.0,  1.0, 1.0, 1.0,
         1.0, -1.0, 1.0, 0.0,
    ]
    ))


    # シェーダープログラム
    program = ctx.program(
        vertex_shader="""
        #version 330

        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 uv;

        void main() {
            gl_Position = vec4(in_pos, 0.999, 1.0);
            uv = in_uv;
        }
        """,
        fragment_shader="""
        #version 330

        uniform sampler2D texture0;
        out vec4 fragColor;
        in vec2 uv;

        void main() {
            fragColor = texture(texture0, uv);
        }
        """,
    )

    # Vertex Array Object の作成
    vao = ctx.vertex_array(program, [(vbo, '2f 2f', 'in_pos', 'in_uv')])

    # テクスチャ画像の読み込み
    textureimg = np.zeros([TEXTURESIZE, TEXTURESIZE, 3],dtype=np.uint8)
    texture = ctx.texture(textureimg.shape[:2], 3, textureimg.tobytes())
    texture.build_mipmaps()
    texture.use()

    print(textureimg.shape)

    return vao, texture


UPDATED = False
VERTSLIST = [None,] * MAXPEOPLE
DETECTION = [True,] * MAXPEOPLE
IMG = None

#Used to initialize VBO
FACES = None
VERTS = None
COLORLIST = [None,]

from threading import Thread, Lock

mutex = Lock()

def Update(vertslist, detection, img=None):
    global UPDATED, VERTSLIST, DETECTION, IMG
    mutex.acquire()
    VERTSLIST = vertslist
    DETECTION = detection
    IMG = img
    UPDATED = True
    mutex.release()


class SimpleColorTriangle(Example):
    gl_version = (3, 3)
    #aspect_ratio = 1 / 1
    title = "GL Viewer"

    def __init__(self, OfflineCTX=None, **kwargs):
        assert(VERTS is not None)
        assert(COLORLIST is not None)
        if OfflineCTX is not None:
            self.ctx = OfflineCTX
        else:
            super().__init__(**kwargs)

        self.window_size = (GLW, GLH)
        self.aspect_ratio = GLW / GLH
        print(self.window_size)
        #assert(GLW != 512)

        self.ctx.enable(moderngl.DEPTH_TEST)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_vert;
                in vec3 in_color;
                out vec3 v_color;    // Goes to the fragment shader
                void main() {
                    gl_Position = vec4(in_vert, 1.0);
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    // We're not interested in changing the alpha value
                    f_color = vec4(v_color, 1.0);
                }
            ''',
        )

        self.ibo = self.ctx.buffer(
            FACES.astype('i4').tobytes())

        self.vbos = []
        self.vaos = []

        for i in range(MAXPEOPLE):
            vbo = self.ctx.buffer(VERTS * 0)
            cbo = self.ctx.buffer(COLORLIST[i])
            # We control the 'in_vert' and `in_color' variables
            vao = self.ctx.vertex_array(
                self.prog,
                [
                    # Map in_vert to the first 2 floats
                    # Map in_color to the next 3 floats
                    (vbo, '3f', 'in_vert'),
                    (cbo, '3f', 'in_color')
                ],
                self.ibo,
            )

            self.vbos.append(vbo)
            self.vaos.append(vao)

        self.vaobg, self.texture = TextureShader(self.ctx)
        self.fbo = None

    def render0(self):
        global UPDATED
        
        if UPDATED:
            mutex.acquire()
            if IMG is not None:
                self.texture.write((IMG[::-1,:,::-1]).tobytes())

            #print(DETECTION)
            for i in range(MAXPEOPLE):
                if DETECTION[i]:
                    #print(VERTSLIST[i][0])
                    if VERTSLIST[i] is not None:
                        self.vbos[i].write(VERTSLIST[i])
            UPDATED = False
            mutex.release()

        

        self.vaobg.render(mode=moderngl.TRIANGLE_STRIP)
        for i in range(MAXPEOPLE):
            if DETECTION[i]:
                self.vaos[i].render()

    def offlinerender(self):
        if self.fbo is None:
            self.fbo = self.ctx.simple_framebuffer((GLW, GLH))
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.render0()

        image = np.frombuffer(self.fbo.read(), dtype=np.uint8)
        image = image.reshape([GLH, GLW, -1])[...,:3]
        image = image[::-1,:,::-1]

        #print(fbo.size, image.shape)
        #Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()
        return image

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.render0()


import time
#SimpleColorTriangle.run()

import threading

GLTHREADRUNNNING = False
def Run():
    global GLTHREADRUNNNING
    if not GLTHREADRUNNNING:
        GLTHREADRUNNNING = True
        t1 = threading.Thread(target=SimpleColorTriangle.run, args=())
        t1.daemon = True
        t1.start()
        
        
        print("GLTHREADRUNNNING")


