import argparse
from Track2.utils import *
from keras.models import load_model
import gc
from keras.preprocessing import image

def get_test(path,res):
    test=[]
    f = open(path, "r+", encoding='utf-8')
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

def read_image(path):
    img = image.load_img(path, target_size=(112, 112))
    img = np.array(img).astype(np.float)
    return img

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
            datas.append([now[0],now[1],now[2]])
            labels.append(int(now[4]))
            classes.append(now[3])
        father_image = np.array([read_image(x[0]) for x in datas])
        mother_image = np.array([read_image(x[1]) for x in datas])
        child_image= np.array([read_image(x[2]) for x in datas])
        yield [father_image, mother_image,child_image], np.array(labels),classes,batch_list
        start=end
        if start == total:
            yield None,None,None,None
        gc.collect()

def test(args):
    model_path = args.save_path
    sample_path = args.sample
    batch_size = args.batch_size
    log_path = args.log_path
    threshold = args.threshold
    model = load_model(model_path)
    res = {
        'FMD': [0, 0],
        'FMS': [0, 0],
        'avg': [0, 0]
    }
    test_samples = get_test(os.path.join(sample_path,"test.txt"), res)
    for imgs, labels, classes,batch_list in gen(test_samples, batch_size):
        if imgs is not None:
            father_em,mother_em,child_em=model(imgs)
            pred=(compute_cosine(father_em, child_em)*0.5+
                  compute_cosine(mother_em,child_em)*0.5).numpy().tolist()
            for i in range(len(pred)):
                if pred[i] >= threshold:
                    p = 1
                else:
                    p = 0
                if p == labels[i]:
                    res['avg'][1] +=  1
                    res[classes[i]][1] += 1
        else:
            break
    for key in res:
        mylog(key, ':', res[key][1] / res[key][0],path=log_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="test  accuracy")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--threshold", type=float, help=" threshold ")
    parser.add_argument("--batch_size", type=int, default=12, help="batch size default 12")
    parser.add_argument("--log_path", type=str, default="./log.txt", help="log path default log.txt ")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(100)
    test(args)
