import argparse
import os
from Track2.utils import *
from Track2.dataset import FIW
from Track2.models import get_model,get_save_model
import gc
from sklearn.metrics import roc_curve,auc
from keras import backend as K
import Track2.losses



def training(args):
    batch_size = args.batch_size
    val_batch_size = 25
    epochs = args.epochs
    steps_per_epoch = 50
    save_path = args.save_path
    Track2.losses.beta = args.beta
    log_path = args.log_path


    train_dataset = FIW(os.path.join(args.sample, "train.txt"))
    val_dataset = FIW(os.path.join(args.sample, "val.txt"))

    model = get_model()
    save_model = get_save_model(model)

    max_auc = 0
    for epoch_i in range(epochs):
        mylog("\n*************", path=log_path)
        mylog('epoch ' + str(epoch_i + 1), path=log_path)

        n = 0
        compare_loss_epoch = 0
        while True:
            father,mother,child,label=train_dataset.get_batch_pairs_image(batch_size)
            loss=model.train_on_batch([father,mother,child],tf.zeros((batch_size)))
            compare_loss_epoch += loss
            n = n + 1
            if n == steps_per_epoch:
                break

        mylog("compare_loss:" + "%.6f" % (compare_loss_epoch / steps_per_epoch), path=log_path)
        auc ,threshold = val_model(save_model, val_dataset ,batch_size=val_batch_size)
        mylog("auc is %.6f "% auc,path=log_path)
        if max_auc < auc:
            mylog("auc improve from :" + "%.6f" % max_auc + " to %.6f" % auc,path=log_path)
            mylog("threshold is : %.8f" % threshold, path=log_path)
            max_auc=auc
            mylog("save model " + save_path,path=log_path)
            save_model.save(save_path)
        else:
            mylog("auc did not improve from %.6f" % float(max_auc),path=log_path)
        gc.collect()
        K.clear_session()


def val_model(model, fiw, batch_size=25):
    y_true = []
    y_pred = []
    for father, mother,child,labels in fiw.get_val_on_batch(fiw.sample_list, batch_size):
        if father is not None:
            father_em,mother_em,child_em=model([father,mother,child])
            y_pred.extend((compute_cosine
                           (father_em, child_em)*0.5+compute_cosine(mother_em,child_em)*0.5).numpy().tolist())
            y_true.extend(labels)
        else:
            break
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_pred)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds_keras[maxindex]
    return auc(fpr,tpr),threshold



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=15, help="batch size default 15")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--epochs", type=int, default=30, help="epochs number default 30")
    parser.add_argument("--beta", default=0.08, type=float, help="beta default 0.08")
    parser.add_argument("--log_path", default="./log.txt", type=str, help="log path default log.txt")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(seed=100)
    training(args)
