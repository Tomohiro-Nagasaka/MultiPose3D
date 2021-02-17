import pickle
from chumpy import Ch
import numpy as np

class Dummy:
    pass

class MyUnpickler(pickle._Unpickler):
    def find_class(self, module, name):
        print(module, name)
        return super().find_class(module, name)
        
        # try:
        #     return super().find_class(module, name)
        # #except AttributeError:
        # except Exception:
        #     return Dummy


def LoadSMPL(model_path):
    with open(model_path, 'rb') as f:
        #params = pickle.load(f,encoding="latin1")
        params = MyUnpickler(f,encoding="latin1").load()
    
    return params
    
def At(A, B, BatchDims=0):
    LA = len(A.shape)
    LB = len(B.shape)

    A2 = A.reshape(list(A.shape) + [1,] * (LB - 1 - BatchDims))
    B2 = B.reshape(list(B.shape)[:BatchDims] + [1,] * (LA - 1 - BatchDims) + list(B.shape)[BatchDims:])

    AB = A2*B2
    AB = np.sum(AB, axis=LA -1)
    return AB

class SMPLModel():
    def __init__(self, model_path, BatchDims = 0, OnlyJoints=True):

        self.BatchDims = BatchDims

        params = LoadSMPL(model_path)

        print(list(params.keys()))

        self.OnlyJoints = OnlyJoints
        self.J_regressor = np.array(params['J_regressor'].toarray(), dtype=np.float32)

        self.weights = np.array(params['weights'], dtype=np.float32)
        self.posedirs = np.array(params['posedirs'], dtype=np.float32)
        self.v_template = np.array(params['v_template'], dtype=np.float32)
        try:
            self.shapedirs = np.array(params['shapedirs'], dtype=np.float32)
        except Exception:
            self.shapedirs = np.array(params['shapedirs'].x, dtype=np.float32)

        if self.OnlyJoints:
            for i in range(BatchDims):
                self.J_regressor = self.J_regressor[np.newaxis]
            self.weights = At(self.J_regressor ,self.weights)
            self.posedirs = At(self.J_regressor ,self.posedirs)
            self.v_template = At(self.J_regressor ,self.v_template)
            self.shapedirs = At(self.J_regressor ,self.shapedirs)
        else:
            for i in range(BatchDims):
                self.J_regressor = self.J_regressor[np.newaxis]
                self.weights = self.weights[np.newaxis]
                self.posedirs = self.posedirs[np.newaxis]
                self.v_template = self.v_template[np.newaxis]
                self.shapedirs = self.shapedirs[np.newaxis]

        self.faces = params['f']