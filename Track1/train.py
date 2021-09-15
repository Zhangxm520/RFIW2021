import gc
from sklearn.metrics import roc_curve,auc
from Track1.dataset import *
from Track1.models import *
from torch.optim import SGD
from Track1.losses import *
import argparse


def training(args):

    batch_size=args.batch_size
    val_batch_size=args.batch_size
    epochs=args.epochs
    steps_per_epoch=50
    save_path=args.save_path
    beta=args.beta
    log_path=args.log_path


    train_dataset=FIW(os.path.join(args.sample,"train_sort.txt"))
    val_dataset=FIW(os.path.join(args.sample,"val_choose.txt"))

    train_loader=DataLoader(train_dataset,batch_size=batch_size,num_workers=1,pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=1, pin_memory=False)

    model=Net().cuda()

    optimizer_model = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    max_auc=0.0

    for epoch_i in range(epochs):
        mylog("\n*************",path=log_path)
        mylog('epoch ' + str(epoch_i+ 1),path=log_path)
        compare_loss_epoch = 0

        model.train()

        for index_i, data in enumerate(train_loader):
            image1,image2,labels =data

            e1,e2,x1,x2= model([image1,image2])

            loss = compare_loss(x1,x2,beta=beta)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            compare_loss_epoch += loss.item()
            
            if (index_i+1)==steps_per_epoch:
                break

        use_sample=(epoch_i+1)*batch_size*steps_per_epoch
        train_dataset.set_bias(use_sample)

        mylog("compare_loss:" + "%.6f" % (compare_loss_epoch / steps_per_epoch),path=log_path)
        model.eval()
        with torch.no_grad():
            auc = val_model(model, val_loader)
        mylog("auc is %.6f "% auc,path=log_path)
        if max_auc < auc:
            mylog("auc improve from :" + "%.6f" % max_auc + " to %.6f" % auc,path=log_path)
            max_auc=auc
            mylog("save model " + save_path,path=log_path)
            save_model(model,save_path)
        else:
            mylog("auc did not improve from %.6f" % float(max_auc),path=log_path)

def save_model(model,path):
    torch.save(model.state_dict(),path)

def val_model(model, val_loader):
    y_true = []
    y_pred = []
    for img1, img2, labels in val_loader:
        e1,e2,_,_=model([img1.cuda(),img2.cuda()])
        y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())
        y_true.extend(labels.cpu().detach().numpy().tolist())
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr,tpr)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=25,help="batch size default 25")
    parser.add_argument( "--sample", type=str, help="sample root")
    parser.add_argument( "--save_path",  type=str, help="model save path")
    parser.add_argument( "--epochs", type=int,default=80, help="epochs number default 80")
    parser.add_argument( "--beta", default=0.08, type=float, help="beta default 0.08")
    parser.add_argument( "--log_path", default="./log.txt", type=str, help="log path default log.txt")
    parser.add_argument( "--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    training(args)
