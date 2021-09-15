import torch
import os
import random
import numpy as np

def np2tensor(arrays,device='gpu'):
    tensor=torch.from_numpy(arrays).type(torch.float)
    return tensor.cuda() if device=='gpu' else tensor

def mylog(*t,path = 'log.txt'):
    t=" ".join([str(now) for now in t])
    print(t)
    if os.path.isfile(path) == False:
        f = open(path, 'w+')
    else:
        f = open(path, 'a')
    f.write(t + '\n')
    f.close()

def set_seed(seed):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
