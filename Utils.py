import numpy as np

def IsNumpyArray(A):
    return isinstance(A, np.ndarray) or isinstance(A, float) or isinstance(A, np.float32)  


def GetBackend(A):
    if(IsNumpyArray(A)):
        K = np
    else:
        K = tf
    return K

def QuatToRot(q):
    K = GetBackend(q)
    norm = K.linalg.norm(q, axis=-1, keepdims=True)
    q = q / norm
    
    tmp=K.stack([1 - 2.*(q[...,2]*q[...,2] + q[...,3]*q[...,3]), 2*(q[...,1]*q[...,2] - q[...,3]*q[...,0]), 2*(q[...,1]*q[...,3] + q[...,2]*q[...,0]),
    2*(q[...,1]*q[...,2] + q[...,3]*q[...,0]), 1 - 2.*(q[...,1]*q[...,1] + q[...,3]*q[...,3]), 2*(q[...,2]*q[...,3] - q[...,1]*q[...,0]),
    2*(q[...,1]*q[...,3] - q[...,2]*q[...,0]), 2*(q[...,2]*q[...,3] + q[...,1]*q[...,0]), 1 - 2.*(q[...,1]*q[...,1] + q[...,2]*q[...,2])],axis=-1)
    
    if IsNumpyArray(q):
        return K.reshape(tmp, [*q.shape[:-1], 3, 3])   
    else:
        shape = K.shape(q)
        shape = K.concat([shape[:-1], [3, 3]], axis=0)
        return K.reshape(tmp, shape)

    #return K.reshape(tmp, [*q.shape[:-1], 3, 3])

#WXYZ
def RotToQuat(R):
    K = GetBackend(R)
    tmp=K.stack([

        R[...,0,0] + R[...,1,1] + R[...,2,2],
        -R[...,1,2] + R[...,2,1],
        -R[...,2,0] + R[...,0,2],
        -R[...,0,1] + R[...,1,0],
        
        -R[...,1,2] + R[...,2,1],
        R[...,0,0] - R[...,1,1] - R[...,2,2], 
        R[...,1,0] + R[...,0,1],
        R[...,2,0] + R[...,0,2],
        
        -R[...,2,0] + R[...,0,2],
        R[...,1,0] + R[...,0,1],
        -R[...,0,0] + R[...,1,1] - R[...,2,2], 
        R[...,2,1] + R[...,1,2],
        
        -R[...,0,1] + R[...,1,0],
        R[...,2,0] + R[...,0,2],
        R[...,2,1] + R[...,1,2],
        -R[...,0,0] - R[...,1,1] + R[...,2,2],    
          

    ],axis=-1)
    
    if IsNumpyArray(R):
        A = K.reshape(tmp, [*R.shape[:-2], 4, 4])  
    else:
        shape = K.shape(R)
        shape = K.concat([shape[:-2], [4, 4]], axis=0)
        A = K.reshape(tmp, shape)
    
    w, v = K.linalg.eigh(A)
    return v[...,-1]

