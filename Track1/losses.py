import torch
import numpy as np
from Track1.utils import *


def contrastive_loss(x1,x2,beta=0.08):
    x1x2=torch.cat([x1,x2],dim=0)
    x2x1=torch.cat([x2,x1],dim=0)

    cosine_mat=torch.cosine_similarity(torch.unsqueeze(x1x2,dim=1),
                                       torch.unsqueeze(x1x2,dim=0),dim=2)/beta
    mask=torch.eye(x1.size(0))
    mask=torch.cat([mask,mask],dim=0)
    mask = 1.0-torch.cat([mask, mask], dim=1)

    numerators = torch.exp(torch.cosine_similarity(x1x2,x2x1,dim=1)/beta)
    denominators=torch.sum(torch.exp(cosine_mat)*mask,dim=1)
    return -torch.mean(torch.log(numerators/denominators),dim=0)
