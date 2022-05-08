import torch
from Track1.torch_resnet101 import *


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder=KitModel("./kit_resnet101.pkl")

        self.projection=nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight -0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        img1,img2=imgs
        embeding1 ,embeding2= self.encoder(img1),self.encoder(img2)
        pro1 ,pro2=self.projection(embeding1),self.projection(embeding2)
        return embeding1,embeding2,pro1,pro2

