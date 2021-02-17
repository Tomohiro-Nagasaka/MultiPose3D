
#%%
VIBEFOLDER = r'F:\Git2\VIBE\data\vibe_data/'
filename = VIBEFOLDER + 'SMPL_NEUTRAL.pkl'
import pickle
#import chumpy
with open(filename, 'rb') as f:
    params = pickle.load(f,encoding="latin1")

print(params)
sd = params['shapedirs']
print(type(sd))
print(sd.x)
#%%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import numpy as np
import pickle
VIBEFOLDER = r'F:\Git2\VIBE\data\vibe_data/'
filename = VIBEFOLDER + 'SMPL_NEUTRAL.pkl'

class Dummy:
    pass

class MyUnpickler(pickle._Unpickler):
    def find_class(self, module, name):
        print(module, name)
        
        
        #return super().find_class(module, name)
        try:
            return super().find_class(module, name)
        #except AttributeError:
        except Exception:
            #return super().find_class("numpy", "ndarray")
            return Dummy


with open(filename, 'rb') as f:
    params = MyUnpickler(f,encoding="latin1").load()

sd = params['shapedirs']
print(type(sd))
print(sd.x)

#%%

import sys
import pickle



class MyUnpickler(pickle._Unpickler):
    def find_class(self, module, name): # from the pickle module code but with a try
        # Subclasses may override this. # we are doing it right now...
        try:
            if self.proto < 3 and self.fix_imports:
                if (module, name) in _compat_pickle.NAME_MAPPING:
                    module, name = _compat_pickle.NAME_MAPPING[(module, name)]
                elif module in _compat_pickle.IMPORT_MAPPING:
                    module = _compat_pickle.IMPORT_MAPPING[module]
            __import__(module, level=0)
            if self.proto >= 4:
                return _getattribute(sys.modules[module], name)[0]
            else:
                return getattr(sys.modules[module], name)
        except AttributeError:
            return Dummy

# edit: as per Ben suggestion an even simpler subclass can be used
# instead of the above

class MyUnpickler2(pickle._Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except AttributeError:
            return Dummy

class C:
    pass

c1 = C()

with open('data1.dat', 'wb') as f:
    pickle.dump(c1,f)

del C # simulate the missing class

#%%
with open(VIBEFOLDER + 'SMPL_NEUTRAL.pkl', 'rb') as f:
    unpickler = MyUnpickler(f) # or MyUnpickler2(f)
    c1 = unpickler.load()

print(c1) # got a Dummy object because of missing class