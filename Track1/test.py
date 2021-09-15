import argparse
import numpy as np
import gc
from Track1.models import Net
import sys
import torch
from keras.preprocessing import image
import os
from Track1.utils import *

def baseline_model(model_path):
    model = Net().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_test(sample_path,res):
    test_file_path = os.path.join(sample_path,"test.txt")
    test=[]
    f = open(test_file_path, "r+", encoding='utf-8')
    while True:
        line=f.readline().replace('\n','')
        if not line:
            break
        else:
            test.append(line.split(' '))
    f.close()
    res['avg'][0]=len(test)
    for now in test:
        res[now[3]][0]+=1
    return test


def gen(list_tuples, batch_size):
    total=len(list_tuples)
    start=0
    while True:
        if start+batch_size<total:
            end=start+batch_size
        else:
            end=total
        batch_list=list_tuples[start:end]
        datas=[]
        labels=[]
        classes=[]
        for now in batch_list:
            datas.append([now[1],now[2]])
            labels.append(int(now[4]))
            classes.append(now[3])
        X1 = np.array([read_image(x[0]) for x in datas])
        X2 = np.array([read_image(x[1]) for x in datas])
        yield X1, X2, labels,classes,batch_list
        start=end
        if start == total:
            yield None,None,None,None,None
        gc.collect()


def read_image(path):
    img = image.load_img(path, target_size=(112, 112))
    img = np.array(img).astype(np.float)
    return np.transpose(img, (2, 0, 1))


def test(args):
    model_path = args.save_path
    sample_path = args.sample
    batch_size = args.batch_size
    log_path = args.log_path
    threshold = args.threshold
    model = baseline_model(model_path)
    classes = [
        'bb', 'ss', 'sibs', 'fd', 'md', 'fs', 'ms', 'gfgd', 'gmgd', 'gfgs', 'gmgs', 'avg'
    ]
    res={}
    for n in classes:
        res[n]=[0,0]
    test_samples = get_test(sample_path, res)
    with torch.no_grad():
        for img1, img2, labels, classes, batch_list in gen(test_samples, batch_size):
            if img1 is not None:
                img1 = torch.from_numpy(img1).type(torch.float).cuda()
                img2 = torch.from_numpy(img2).type(torch.float).cuda()
                em1, em2, _, _ = model([img1, img2])
                pred = torch.cosine_similarity(em1, em2, dim=1).cpu().detach().numpy().tolist()
                for i in range(len(pred)):
                    if pred[i] >= threshold:
                        p = 1
                    else:
                        p = 0
                    if p == labels[i]:
                        res['avg'][1] += 1
                        res[classes[i]][1] += 1
            else:
                break
    for key in res:
        mylog(key, ':', res[key][1] / res[key][0], path=log_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="test  accuracy")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--threshold", type=float, help=" threshold ")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size default 40")
    parser.add_argument("--log_path", type=str, default="./log.txt", help="log path default log.txt ")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(100)
    test(args)
