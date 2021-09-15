import gc
from sklearn.metrics import roc_curve,auc
from Track1.dataset import *
from Track1.models import *
from Track1.utils import *
import Track1.torch_resnet101
import torch
from torch.optim import SGD
from Track1.losses import *
import sys
import argparse

def find(args):
    path=args.save_path
    batch_size=args.batch_size
    log_path=args.log_path
    val_dataset = FIW(os.path.join(args.sample,"val.txt"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, pin_memory=False)
    model = Net().cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        auc ,threshold = val_model(model, val_loader)
    mylog("auc : ",auc,path=log_path)
    mylog("threshold :" ,threshold,path=log_path)


def val_model(model, val_loader):
    y_true = []
    y_pred = []
    for img1, img2, labels in val_loader:
        e1,e2,_,_=model([img1.cuda(),img2.cuda()])
        y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())
        y_true.extend(labels.cpu().detach().numpy().tolist())
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_pred)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds_keras[maxindex]
    return auc(fpr,tpr),threshold


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="find threshold")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size default 40")
    parser.add_argument("--log_path", type=str, default="./log.txt",help="log path default log.txt ")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(100)
    find(args)
