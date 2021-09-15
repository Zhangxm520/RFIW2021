import torch
from Track3.torch_resnet101 import *


class Net(torch.nn.Module):
    def __init__(self,path):
        super(Net, self).__init__()
        self.encoder=KitModel("./tch/kit_resnet101.pkl")
        self.load_weights(path)

    def load_weights(self,path):
        state=torch.load(path)
        del_keys=[]
        for key in state.keys():
            if key.find("encoder")==-1:
                del_keys.append(key)
        for k in del_keys:
            del state[k]
        self.load_state_dict(state,strict=False)

    def forward(self, img):
        return self.encoder(img)
